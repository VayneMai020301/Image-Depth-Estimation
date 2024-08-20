from cfg import  np, sys
sys.path.append('/core/')
def image_casting(left, right):

    left = left.astype(np.float32 )
    right = right.astype(np.float32 )
    height , width = left.shape[:2]
    return left, right, height,width