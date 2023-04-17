import os
import gc
import time
import math
import torch
import logging
import pandas as pd
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler

from irnlm.models.utils import save_checkpoint, format_time
from irnlm.models.gpt2.extract_features_gpt2_integral import create_examples, pad_to_max_length
from irnlm.utils import get_timestamp, check_folder


#########################################
############## Base Class ###############
#########################################

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_data(self, dataset_object, set_type):
        """Retrieve data for the `set_type` set."""
        raise NotImplementedError()

    def get_labels(self, dataset_object):
        """Gets the list of labels for this data set."""
        return dataset_object.get_labels()

    def get_data_loader(self, features, batch_size, local_rank):
        """Return data loader object."""
        raise NotImplementedError()


class ModelProcessor(object):
    """Base class for model training/validation and evaluation."""

    def __init__(self, 
                    model=None, 
                    optimizer=None, 
                    tokenizer=None, 
                    scheduler=None, 
                    device=None, 
                    metric_name=None, 
                    nb_epochs=3, 
                    use_output_mask=False, 
                    context_size=None,
                    nb_steps=None,
                    nb_checkpoints=24):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.device = device
        self.nb_checkpoints = nb_checkpoints
        self.nb_epochs = nb_epochs
        self.metric_name = metric_name
        self.use_output_mask = use_output_mask
        self.nb_steps=nb_steps
        self.context_size = context_size
        
    def attention_mask_from_inputs(self, input_ids, context_size):
        """Compute the attention mask for each input_ids batch.
        """
        if context_size is None:
            attention_mask = torch.ones(input_ids.size()).to(torch.int64)
        else:
            attention_mask =  torch.diag_embed(torch.tensor([0 for x in range(input_ids.size(-1))])) 
            for i in range(min(input_ids.size(-1), context_size + 1)): # Adding 1 so that context==0 is only the current word
                attention_mask = torch.add(attention_mask, torch.diag_embed(torch.tensor([1 for x in range(input_ids.size(-1) - i)]), offset=-i))
            attention_mask = attention_mask.unsqueeze(0).repeat(input_ids.size(0), 1, 1).to(torch.int64)
        return attention_mask


    #########################################
    ########### Training functions ##########
    #########################################

    def training_step(self, batch, total_train_loss):
        """ Compute a training step in a model training.
        Arguments:
            - batch:
            - total_train_loss:
        Returns:
            - total_train_loss: (float) accumulated loss from the batch 
            and the input
        """
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        ##   [1]: attention masks
        #   [1]: token type ids 
        #   [2]: labels 
        input_ids = batch[0].to(torch.int64).to(self.device)
        #attention_mask = batch[1].to(torch.int64).to(self.device)
        #attention_mask = self.attention_mask_from_inputs(batch[0], self.context_size).to(torch.int64).to(self.device)
        attention_mask = None #torch.ones(batch[0].size()).to(torch.int64).to(self.device)
        token_type_ids = None #torch.zeros(input_ids.size()).to(torch.int64).to(self.device) #batch[1].to(torch.int64).to(self.device)
        labels_ids = input_ids.clone() #batch[2].to(torch.int64).to(self.device)

        self.model.zero_grad()        

        outputs = self.model(input_ids, 
                            token_type_ids=token_type_ids, 
                            attention_mask=attention_mask, 
                            labels=labels_ids)
        loss = outputs[0] 
        # The `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        self.optimizer.step()
        # Update the learning rate.
        self.scheduler.step()
        return total_train_loss

    def train(self, data_processor, train_data_paths, validation_features_paths, output_dir, parameters, start_at_dataloader=0):
        """ Train a model with evaluation at each step, given an optimizer, scheduler, device and train and 
        validation data loaders.
        Returns loss statistics from training and evaluations.
        """
        special_token_beg = self.tokenizer.bos_token
        special_token_end = self.tokenizer.eos_token
        special_token_beg_ids = self.tokenizer(self.tokenizer.bos_token)['input_ids'][0]
        special_token_end_ids = self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
        try:
            special_token_pad = self.tokenizer.pad_token
            special_token_pad_ids = self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]
            space = None
        except ValueError:
            special_token_pad = None
            special_token_pad_ids = None
            space = 220
            
        training_stats = []
        validation_stats = []
        logging.basicConfig(filename=os.path.join(parameters['output_dir'], parameters['log_file']), filemode='w+', level=logging.INFO)

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        checkpoints_index = parameters['init_checkpoints']
        
        #for epoch_i in range(parameters['start_epoch'], self.nb_epochs):
        for epoch_i in range(0, self.nb_epochs):
            print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.nb_epochs))
            logging.info('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.nb_epochs))
            logging.info('Training...')
            print('Training...')
            if epoch_i >= parameters['start_epoch']:
                if epoch_i==0 and start_at_dataloader==0:
                    logging.info("Saving model at the start of epoch {} to {}...".format(epoch_i, os.path.join(output_dir, f'start-epoch-{epoch_i}')))
                    save_checkpoint(self.model, self.tokenizer, output_dir, f'start-epoch-{epoch_i}')
                    logging.info("\tDone.")

                # Measure how long the training epoch takes.
                t0 = time.time()
                # Reset the total loss for this epoch.
                total_train_loss = 0
                nb_batchs_done = 0
                if parameters['do_train']:
                    # Put the model into training mode. Don't be mislead--the call to 
                    # `train` just changes the *mode*, it doesn't *perform* the training.
                    # `dropout` and `batchnorm` layers behave differently during training
                    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
                    self.model.train()

                    for split_index, batch_path in enumerate(train_data_paths):
                        if (split_index >= start_at_dataloader) or (start_at_dataloader+1==len(train_data_paths)):
                            start_at_dataloader = 0
                            logging.info(f"[{get_timestamp()}] - Creating training data loader for split {split_index}..")
                            data = data_processor.load_object(batch_path)
                            #print('------- Data augmentation -------')
                            #examples = [data_processor.create_examples(data[i:i + self.context_size + 2]) for i, _ in tqdm(enumerate(data[:-self.context_size -2]))]
                            n = len(data)
                            print('------- No data augmentation -------')
                            #examples_ids, examples_masks = list(zip(*[data_processor.create_examples(data[i*self.context_size:min((i+1)*self.context_size + 2, n)]) for i in tqdm(range(n//self.context_size))]))
                            if self.context_size==0:
                                examples_ids = [create_examples(
                                                    data[2*i:2*(i+1)],
                                                    parameters['max_length'],
                                                    space=space, 
                                                    special_token_beg=special_token_beg_ids, 
                                                    special_token_end=special_token_end_ids, 
                                                    special_token_pad=special_token_pad_ids
                                                ) for i in tqdm(range(n//2))]
                            else:
                                examples_ids = [create_examples(
                                                    data[i*self.context_size:min((i+1)*self.context_size + 2, n)],
                                                    parameters['max_length'],
                                                    space=space, 
                                                    special_token_beg=special_token_beg_ids, 
                                                    special_token_end=special_token_end_ids, 
                                                    special_token_pad=special_token_pad_ids
                                                ) for i in tqdm(range(n//self.context_size))]
                            features = [torch.FloatTensor(example).unsqueeze(0).to(torch.int64) for example in tqdm(examples_ids)]
                            #masks = [torch.FloatTensor(mask).unsqueeze(0).to(torch.int64) for mask in tqdm(examples_masks)]
                            input_ids = torch.cat(features, dim=0)
                            #attention_masks = torch.cat(masks, dim=0)
                            data = TensorDataset(input_ids) #, attention_masks)
                            sampler = RandomSampler(data)
                            dataloader = DataLoader(data, sampler=sampler, batch_size=parameters['batch_size'])
                            # Cleaning
                            del data
                            del examples_ids
                            del features
                            del input_ids
                            
                            logging.info(f"[{get_timestamp()}] - \tDone.")
                            save_step = max(1, self.nb_epochs * len(dataloader) * len(train_data_paths) // self.nb_checkpoints)

                            # For each batch of training data...
                            for step, batch in enumerate(dataloader):
                                step += nb_batchs_done

                                # Save model weights to have a given number of checkpoints at the end
                                if step != 0 and step % save_step == 0:
                                    save_checkpoint(self.model, self.tokenizer, output_dir, 'checkpoint_' + str(checkpoints_index))
                                    checkpoints_index += 1
                                # Progress update every 50 batches.
                                if step % min(50, save_step) == 0 and not step == 0:
                                    # Calculate elapsed time in minutes.
                                    elapsed = format_time(time.time() - t0)
                                    # Report progress.
                                    lr = vars(self.optimizer)['param_groups'][0]['lr']
                                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f}e-5 | ms/batch {} | '
                                    'loss {:5.2f} | ppl {:8.2f}'.format(epoch_i, step, self.nb_steps//parameters['batch_size'], lr*10**5, elapsed, total_train_loss-tmp, math.exp(total_train_loss-tmp))) # / :5.2f
                                    logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f}e-5 | ms/batch {} | '
                                    'loss {:5.2f} | ppl {:8.2f}'.format(epoch_i, step, self.nb_steps//parameters['batch_size'], lr*10**5, elapsed, total_train_loss-tmp, math.exp(total_train_loss-tmp)))
                                tmp = total_train_loss if step>0 else 0
                                total_train_loss = self.training_step(batch, total_train_loss)
                            nb_batchs_done += len(dataloader)
                            # Cleaning
                            del dataloader
                            gc.collect()
                            # Saving
                            logging.info("Saving model at the end of DataLoader {} to {}...".format(split_index, os.path.join(output_dir, f'end-epoch-{epoch_i}_split-{split_index}')))
                            save_checkpoint(self.model, self.tokenizer, output_dir, f'end-epoch-{epoch_i}_split-{split_index}')
                            logging.info("\tDone.")
                        else:
                            print(f"Skipping dataloader #{split_index}.")
                            logging.info(f"Skipping dataloader #{split_index}.")
                    # Calculate the average loss over all of the batches.
                    avg_train_loss = total_train_loss / nb_batchs_done           
                    # Measure how long this epoch took.
                    training_time = format_time(time.time() - t0)

                    print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
                    logging.info("\n  Average training loss: {0:.2f}".format(avg_train_loss))
                    logging.info("Saving model at the end of epoch {} to {}...".format(epoch_i, os.path.join(output_dir, f'end-epoch-{epoch_i}')))
                    save_checkpoint(self.model, self.tokenizer, output_dir, f'end-epoch-{epoch_i}')
                    logging.info("\tDone.")
                    logging.info("  Training epoch took: {:}".format(training_time))
                    print("  Training epoch took: {:}".format(training_time))
                    # Record all statistics from this epoch.
                    training_stats.append(
                        {
                            'epoch': epoch_i + 1,
                            'Training Loss': avg_train_loss,
                            'Training Time': training_time
                        }
                    )
                    df = pd.DataFrame(data=training_stats)
                    df.to_csv(os.path.join(output_dir, 'training_stats.csv'), index=False)
                
                if parameters['do_validation']:
                    val_loss, val_time = self.evaluate(data_processor, validation_features_paths, 'dev', parameters=parameters)
                    # Record all statistics from this epoch.
                    validation_stats.append(
                        {
                            'epoch': epoch_i + 1,
                            'Valid. Loss': val_loss,
                            'Validation Time': val_time,
                        }
                    )
                    df = pd.DataFrame(data=validation_stats)
                    df.to_csv(os.path.join(output_dir, 'validation_stats.csv'), index=False)

                
            
            else:
                logging.info(f"[{get_timestamp()}] - Skipping epoch {epoch_i}...")
                if parameters['do_validation']:
                    try:
                        df = pd.read_csv(os.path.join(output_dir, 'training_stats.csv'))
                        training_stats = df.to_dict('records')
                        df = pd.read_csv(os.path.join(output_dir, 'validation_stats.csv'))
                        validation_stats = df.to_dict('records')
                    except:
                        if (epoch_i+1) >= parameters['start_epoch']:
                            logging.info(f"... but computing the validation loss for previous epoch number {epoch_i}...")
                            val_loss, val_time = self.evaluate(data_processor, validation_features_paths, 'dev', parameters=parameters)
                            validation_stats.append(
                                {
                                    'epoch': epoch_i + 1,
                                    'Valid. Loss': val_loss,
                                    'Validation Time': val_time,
                                }
                            )
                            df = pd.DataFrame(data=validation_stats)
                            df.to_csv(os.path.join(output_dir, 'validation_stats.csv'), index=False)

            
        print("\nTraining complete!")
        logging.info("\nTraining complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        logging.info("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        return df

    def evaluate(self, data_processor, validation_features_paths, set_type, parameters):
        """ Evaluate a model on a validation dataloader.
        """
        special_token_beg = self.tokenizer.bos_token
        special_token_end = self.tokenizer.eos_token
        special_token_beg_ids = self.tokenizer(self.tokenizer.bos_token)['input_ids'][0]
        special_token_end_ids = self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
        try:
            special_token_pad = self.tokenizer.pad_token
            special_token_pad_ids = self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]
            space = None
        except ValueError:
            special_token_pad = None
            special_token_pad_ids = None
            space = 220
            
        print("Creating temporary folder...")
        check_folder(os.path.join(parameters['output_dir'], 'tmp'))
        print("Running Validation...")
        t0 = time.time()
        self.model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        total_active_loss = 0
        nb_eval_steps = 0
        loss_fct = CrossEntropyLoss()
        
        logging.basicConfig(filename=os.path.join(parameters['output_dir'], parameters['log_file']), filemode='w+', level=logging.INFO)

        nb_batchs = 0
        # Evaluate data for one epoch
        for split_index, batch_path in enumerate(validation_features_paths):
            logging.info(f"[{get_timestamp()}] - Creating {set_type} data loader for split {split_index}..")
            
            data = data_processor.load_object(batch_path)
            n = len(data)
            #examples_ids, examples_masks = list(zip(*[data_processor.create_examples(data[i*self.context_size:min((i+1)*self.context_size + 2, n)]) for i in tqdm(range(n//self.context_size))]))
            if self.context_size==0:
                examples_ids = [create_examples(
                                    data[2*i:2*(i+1)],
                                    parameters['max_length'],
                                    space=space, 
                                    special_token_beg=special_token_beg_ids, 
                                    special_token_end=special_token_end_ids, 
                                    special_token_pad=special_token_pad_ids
                                ) for i in tqdm(range(n//2))]
            else:
                examples_ids = [create_examples(
                                    data[i*self.context_size:min((i+1)*self.context_size + 2, n)],
                                    space=space, 
                                    special_token_beg=special_token_beg_ids, 
                                    special_token_end=special_token_end_ids, 
                                    special_token_pad=special_token_pad_ids
                                ) for i in tqdm(range(n//self.context_size))]
            features = [torch.FloatTensor(example).unsqueeze(0).to(torch.int64) for example in tqdm(examples_ids)]
            #masks = [torch.FloatTensor(mask).unsqueeze(0).to(torch.int64) for mask in tqdm(examples_masks)]
            input_ids = torch.cat(features, dim=0)
            #attention_masks = torch.cat(masks, dim=0)
            data = TensorDataset(input_ids)#, attention_masks)
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=parameters['batch_size_eval'])

            logging.info("\tDone.")
            nb_batchs += len(dataloader)
            split_logits = []
            split_label_ids = []
            split_active_loss = []
            
            for batch in tqdm(dataloader):
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                ##   [1]: attention masks   # was removed
                #   [1]: token_type_ids
                #   [2]: labels 
                #   [3]: output_mask (optional)
                input_ids = batch[0].to(torch.int64).to(self.device)
                #attention_mask = batch[1].to(torch.int64).to(self.device)
                #attention_mask = self.attention_mask_from_inputs(batch[0], self.context_size).to(torch.int64).to(self.device)
                attention_mask = torch.ones(batch[0].size()).to(torch.int64).to(self.device)
                token_type_ids = torch.zeros(input_ids.size()).to(torch.int64).to(self.device) #batch[1].to(torch.int64).to(self.device)
                label_ids = input_ids.clone() #batch[2].to(torch.int64).to(self.device)
                label_ids[:, 0] = -100
                label_ids[:, -2] = -100
                label_ids[:, -1] = -100
                #active_loss = (attention_mask == 1)
                
                with torch.no_grad():        
                    # The documentation for the BERT `models` are here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html
                    outputs = self.model(input_ids, 
                                        token_type_ids=token_type_ids, 
                                        attention_mask=attention_mask,
                                        labels=label_ids)
                total_eval_loss += outputs[0].item()
                #logits = outputs[1][:, :-2, :] # we remove the 2 special tokens at the end
                #label_ids = label_ids[..., 1:].contiguous()[:, :-2] # we remove the 2 special tokens at the end
                #if self.use_output_mask:
                #    output_mask = batch[3].numpy()
                #    active_loss = (output_mask == 1)
                #else:
                #    active_loss = np.ones(label_ids.shape)
                #    active_loss[label_ids==50256] = 0
                #    active_loss[label_ids==220] = 0
                #    active_loss = (active_loss == 1)
                #split_active_loss.append(active_loss)
                # Accumulate the validation loss.
                
                ## Shift so that tokens < n predict n
                #shift_logits = logits[active_loss][..., :-1, :].contiguous()
                #shift_labels = label_ids[active_loss][..., 1:].contiguous()
                ## Flatten the tokens
                
                #total_active_loss += loss_fct(logits.view(-1, logits.size(-1)), label_ids.view(-1)).item()

                ## Move logits and labels to CPU
                #split_logits.append(np.argmax(logits.detach().cpu().numpy(), axis=-1)[:, 1:-1])
                #split_label_ids.append(label_ids.to('cpu').numpy())
                
            #logits = np.vstack(split_logits)
            #label_ids = np.vstack(split_label_ids)
            #active_loss = np.vstack(split_active_loss)
        
            #pred_flat = logits[active_loss].flatten()
            #labels_flat = label_ids[active_loss].flatten()
            #logging.info(f"Saving predictions and labels in {os.path.join(parameters['output_dir'], 'tmp')}...")
            #np.save(os.path.join(parameters['output_dir'], 'tmp', f'pred_flat_{split_index}.npy'), pred_flat)
            #np.save(os.path.join(parameters['output_dir'], 'tmp', f'labels_flat_{split_index}.npy'), labels_flat)
            # Cleaning
            #logging.info("Cleaning...")
            #del pred_flat
            #del labels_flat
            #del logits
            #del label_ids
            #del active_loss
            
        # Merging Predicitons and labels
        #logging.info(f"[{get_timestamp()}] - Loading computed predictions and labels & merging...")
        #pred_flat = np.hstack([np.load(os.path.join(parameters['output_dir'], 'tmp', f'pred_flat_{split_index}.npy')) for split_index in range(len(validation_features_paths))])
        #labels_flat = np.hstack([np.load(os.path.join(parameters['output_dir'], 'tmp', f'labels_flat_{split_index}.npy')) for split_index in range(len(validation_features_paths))])

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        #logging.info(f"[{get_timestamp()}] - Computing accuracy...")
        #total_eval_accuracy = Metrics.flat_accuracy(labels_flat, pred_flat)

        # Report results
        #logging.info(f"[{get_timestamp()}] - Computing report...")
        #report = Metrics.report(self.metric_name, labels_flat, pred_flat)
        #print(report)
        # Report the final accuracy for this validation run.
        #avg_val_accuracy = total_eval_accuracy / nb_batchs
        #print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        #logging.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / nb_batchs
        #avg_active_loss = total_active_loss / nb_batchs
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        logging.info("  Validation took: {:}".format(validation_time))
        
        # Cleaning
        #shutil.rmtree(os.path.join(parameters['output_dir'], 'tmp'))
        
        #return avg_val_accuracy, avg_val_loss, validation_time, report
        return avg_val_loss, validation_time
            
