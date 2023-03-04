import numpy as np

def specificity_index(img_sem, img_syn, masker, threshold=4):
    """ Compute the sensitivity score which is the ratio: np.log10(x_sem/s_syn).
    Args:
        - img_sem:
        - img_syn:
    Returns:
        - sensitivity_score: float
    """
    def f(a, b):
        a = max(a, 0)
        b = max(b, 0)
        
        if a+b==0:
            return np.nan
        elif b==0:
            return threshold
        elif a==0:
            return -threshold
        else:
            result = np.log10(a/b)
            if result == 0:
                return -10000
            else: 
                return result
            
        
    a1 = masker.transform(img_sem)
    a2 = masker.transform(img_syn)
    sensitivity_score = masker.inverse_transform(
        np.clip(
            np.vectorize(f, otypes=[float])(a1,a2),
            -threshold,
            threshold
        )
    )

    #sensitivity_score = math_img('1+(img1-img2)/(img1+img2)', img1=img_sem, img2=img_syn)
    return sensitivity_score
