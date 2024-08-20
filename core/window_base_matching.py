from cfg import cv2, np, sys
from cfg import SAVING_NOTIFY, DONE_NOTIFY
sys.path.append('./core/')

from distance import distance_l1
from distance import distance_l2
from distance import cosine_similarity
from utility import image_casting

def total_cost_in_window(kernel_half, max_value,left,right ,total, x,y,j, distance):
    value = 0
    for v in range ( -kernel_half , kernel_half + 1) :
        for u in range (-kernel_half , kernel_half + 1) :
            value = max_value
            if (x + u - j) >= 0:
                if distance == "l1":
                    value = distance_l1(int(left [y + v, x + u]) , int(right[y + v, (x +u) - j]))
                elif distance =="l2":
                    value = distance_l2(int(left [y + v, x + u]) , int(right[y + v, (x +u) - j]))
            total += value
    return total 

def window_based_matching_l1(left_img , right_img , disparity_range , kernel_size = 5 ,save_result = True):
    """
        * Calculate Disparity Map based on Windonw Base method and distance is l1
    """
    left, right , height, width =  image_casting(cv2.imread(left_img, 0), cv2.imread(right_img, 0))

    depth = np.zeros((height, width) , np.uint8 )
    kernel_half = int((kernel_size - 1) / 2)

    scale = 3; max_value = 255 * 9 

    for y in range ( kernel_half , height - kernel_half ) :
        for x in range ( kernel_half , width - kernel_half ) :
            disparity = 0; cost_min = 65534   
            for j in range(disparity_range):
                total = 0
                total = total_cost_in_window(kernel_half,max_value,left, right, total, x, y, j,"l1")
                if total < cost_min :
                    cost_min = total
                    disparity = j
            depth [y, x] = disparity * scale

    if save_result == True :
        print (SAVING_NOTIFY)
    cv2.imwrite ('results/window_based_l1.png', depth )
    cv2.imwrite ('results/window_based_l1_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET ) )
    print(DONE_NOTIFY)
    return depth


def window_based_matching_l2(left_img , right_img , disparity_range , kernel_size =5 ,save_result = True ) :
    """
        * Calculate Disparity Map based on Windonw Base method and distance is l2
    """

    left, right , height, width =  image_casting(cv2.imread(left_img, 0), cv2.imread(right_img, 0))

    depth = np. zeros((height, width), np.uint8 )
    kernel_half = int (( kernel_size - 1) / 2)
    scale = 3;max_value = 255 * 9 
    
    for y in range ( kernel_half , height - kernel_half ) :
        for x in range ( kernel_half , width - kernel_half ) :
            disparity = 0
            cost_min = 65534    
            for j in range ( disparity_range ) :
                total = 0
                total = total_cost_in_window(kernel_half,max_value,left, right, total,x, y, j,"l2")
                if total  <  cost_min :
                    cost_min = total
                    disparity = j
            depth [y, x] = disparity * scale

    if save_result == True :
        print (SAVING_NOTIFY)
    cv2.imwrite ('results/window_based_l2.png', depth )
    cv2.imwrite ('results/window_based_l2_color.png', cv2.applyColorMap(depth , cv2.COLORMAP_JET))
    print (DONE_NOTIFY)

    return depth

def window_based_matching (left_img , right_img , disparity_range , kernel_size =5 ,save_result = True ) :
    """
        * Calculate Disparity Map based on Windonw Base method and distance is cosine distance
    """
    left, right , height, width =  image_casting(cv2.imread(left_img, 0) , cv2.imread(right_img, 0))
    
    depth = np.zeros((height , width ), np.uint8 )
    kernel_half = int ((kernel_size - 1) / 2)
    scale = 3

    for y in range(kernel_half , height - kernel_half ) :
        for x in range(kernel_half , width - kernel_half ) :
            disparity = 0
            cost_optimal = -1

            for j in range(disparity_range ) :
                d = x - j
                cost = -1
                if (d - kernel_half ) > 0:
                    wp = left [(y- kernel_half ) :(y+ kernel_half ) +1 , (x-kernel_half ) :(x+ kernel_half ) +1]
                    wqd = right [(y- kernel_half ) :(y+ kernel_half ) +1 , (d-kernel_half ) :(d+ kernel_half ) +1]
                    wp_flattened = wp.flatten()
                    wqd_flattened = wqd.flatten()
                    cost = cosine_similarity(wp_flattened , wqd_flattened )

                if cost > cost_optimal :
                    cost_optimal = cost
                    disparity = j

            depth [y, x] = disparity * scale

    if save_result == True :
        print (SAVING_NOTIFY)
        cv2.imwrite('results/window_based_cosine_similarity.png ', depth )
        cv2.imwrite('results/window_based_cosine_similarity_color.png ', cv2.applyColorMap(depth, cv2. COLORMAP_JET))

    print (DONE_NOTIFY)    
    return depth