from core.pixel_wise_matching import pixel_wise_matching_l1
from core.pixel_wise_matching import pixel_wise_matching_l2

from core.window_base_matching import window_based_matching_l1
from core.window_base_matching import window_based_matching_l2
from core.window_base_matching import window_based_matching

from cfg import time, argparse

def main() -> int:

    parser = argparse.ArgumentParser(description="The different of programer")
    
    parser.add_argument('-m', type = str, choices=['pixel', 'window'], required=True, default='pixel',
                        help="Select method: pixel, window")
    
    parser.add_argument('-distance',  type = str, choices=['l1', 'l2', 'cosine'],  required=True, default='l1',
                        help="Select Distance: l1, l2, cosine")

    args = parser.parse_args()

    left_img_path   = 'images/left.png'
    right_img_path  = 'images/right.png'
    disparity_range = 16
    kernel_size = 3

    if args.m == "pixel":
        if args.distance =="l1":
            t0 = time.time()
            _ = pixel_wise_matching_l1 (
                left_img_path ,
                right_img_path ,
                disparity_range ,
                save_result = True
            )
            print(f"time pixel_wise_matching_l1: {round(time.time() - t0,2)}(s)")
            return 0 

        elif args.distance =="l2":
            t0 = time.time()
            _ = pixel_wise_matching_l2 (
                left_img_path ,
                right_img_path ,
                disparity_range ,
                save_result = True
            )
            print(f"time pixel_wise_matching_l2: {round(time.time() - t0,2)}(s)")
            return 0 
        
    elif args.m == "window":
        left_img_path   = 'images/Aloe_left_1.png'
        right_img_path  = 'images/Aloe_right_2.png'
        disparity_range = 64
        if args.distance =="l1":
            t0 = time.time()
            _ = window_based_matching_l1 (
                left_img_path ,
                right_img_path ,
                disparity_range ,
                kernel_size = kernel_size,
                save_result = True
                )
            print(f"time window_based_matching_l1: {round(time.time() - t0,2)}(s)")
            return 0 
        elif args.distance =="l2":
            t0 = time.time()
            _ = window_based_matching_l2 (
                left_img_path ,
                right_img_path ,
                disparity_range ,
                kernel_size = kernel_size,
                save_result = True
                )
            print(f"time window_based_matching_l2: {round(time.time() - t0,2)}(s)")
        elif args.distance =="cosine":
            t0 = time.time()
            _ = window_based_matching (
                left_img_path ,
                right_img_path ,
                disparity_range ,
                kernel_size = kernel_size,
                save_result = True
                )
            print(f"time window_based_matching: {round(time.time() - t0,2)}(s)")
    
if __name__ == "__main__":
    main()