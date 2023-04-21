import argparse
from irnlm.data.extract_semantic_features import integral2semantic
from irnlm.utils import write


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract semantic features (content words) from the integral text.')
    parser.add_argument("--text", type=str)
    parser.add_argument("--language", type=str, default='english')
    parser.add_argument("--saving_path", type=str, default="./semantic_features.txt")

    args = parser.parse_args()
    path_integral_text = args.text

    semantic_features = integral2semantic(path_integral_text, language=args.language)
    write(args.saving_path, ' '.join(semantic_features))
