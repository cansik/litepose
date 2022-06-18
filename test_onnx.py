import argparse

import numpy as np
import onnxruntime as rt
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Example inferencing")

    parser.add_argument("--model", help="Path to onnx model", default="models/litepose-auto-xs-crowd-pose.onnx", type=str)
    parser.add_argument("--image", help="Path to image file", default="media/pexels-jansel-ferma-3152430.jpg", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    ih, iw = image.shape[:2]

    session = rt.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    batch_size, channels, width, height = session.get_inputs()[0].shape

    in_frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    in_frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)
    x = np.asarray(in_frame)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # for i in range(x.shape[2]):
    #    x[:, :, i] = x[:, :, i] / (std[i] * 1000.0) - mean[i]

    x = x / 255
    x = x - 0.45

    x = x.astype(np.float32)
    x = x.transpose((2, 0, 1))
    x = x.reshape((batch_size, channels, width, height))

    out = session.run(None, {input_name: x})

    keypoints = []
    for heatmap in out[1][0]:
        w, h = heatmap.shape[:2]
        _, max_val, _, max_indx = cv2.minMaxLoc(heatmap)
        x = int(max_indx[0] / w * iw)
        y = int(max_indx[1] / h * ih)
        keypoints.append((x, y))

    # annotate
    for kp in keypoints:
        cv2.drawMarker(image, kp, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

    cv2.imshow("Result", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
