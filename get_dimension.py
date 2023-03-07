from roboflow import Roboflow
import argparse

def detect(
        conf_thres,
        iou_thres,
        source):
    rf = Roboflow(api_key="lVzyCFPWSZgJDLYsYoxf")
    project = rf.workspace().project("insect_detect_detection")
    model = project.version(3).model    

    print(model.predict(source, confidence=conf_thres, overlap=iou_thres).json())


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    # these can change accordingly
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold') 
    parser.add_argument('--source', type=str, default="/data/images", help='input image')    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    detect(**vars(opt))
    print("--------------Completed--------------------")
