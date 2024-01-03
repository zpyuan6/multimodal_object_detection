from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO('yolov8n.pt')


    model.train(data='yolov8_orig\\uk_pest_dataset_29DEC.yaml', epochs=1)
