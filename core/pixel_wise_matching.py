from cfg import cv2, np, sys
sys.path.append('./core/')
from distance import distance_l1
from distance import distance_l2
from utility import image_casting
from cfg import SAVING_NOTIFY, DONE_NOTIFY

    
def pixel_wise_matching_l1 ( left_img , right_img , disparity_range , save_result = True) :
    """
        * Calculate Disparity Map based on pixel-wise method with distance is L1
        * Save Disparity map(gray scale) and color
        * Return Disparity Map
    """
    
    left, right , height, width =  image_casting(cv2.imread(left_img,0),cv2.imread(right_img, 0))

    depth = np.zeros(( height , width ) , np.uint8)
    scale = 16
    max_value = 255
    for y in range ( height ) :
        for x in range ( width ):
            disparity = 0
            cost_min = max_value
            for j in range( disparity_range ) :
                cost = max_value if (x - j) < 0 else distance_l1 (int(left [y, x]) ,int (right [y, x - j]))

                if cost < cost_min :
                    cost_min = cost
                    disparity = j

            depth [y, x] = disparity * scale

    if save_result == True :
        print(SAVING_NOTIFY)
        cv2.imwrite ('results/pixel_wise_l1.png', depth )
        cv2.imwrite ('results/pixel_wise_l1_color.png', cv2.applyColorMap(depth ,cv2 .COLORMAP_JET))

    print (DONE_NOTIFY)
    return depth

def pixel_wise_matching_l2 ( left_img , right_img , disparity_range , save_result = True) :
    """
        * Calculate Disparity Map based on pixel-wise method with distance is L2
        * Save Disparity map(gray scale) and color
        * Return Disparity Map
    """
    left, right , height, width =  image_casting(cv2.imread(left_img, 0) , cv2.imread(right_img, 0))
    depth = np.zeros(( height , width ) , np.uint8 )
    scale = 16
    max_value = 255
    for y in range ( height ) :
        for x in range ( width ):
            disparity = 0
            cost_min = max_value
            for j in range( disparity_range ) :
                cost = max_value if (x - j) < 0 else distance_l2 (int( left [y, x]) ,int( right [y, x - j]) )
                if cost < cost_min :
                    cost_min = cost
                    disparity = j
        
            depth [y, x] = disparity * scale

    if save_result == True :
        print(SAVING_NOTIFY)
        cv2.imwrite ('results/pixel_wise_l2.png', depth )
        cv2.imwrite ('results/pixel_wise_l2_color.png', cv2.applyColorMap(depth ,cv2.COLORMAP_JET ) )

    print (DONE_NOTIFY)
    return depth

