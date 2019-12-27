import numpy as np
import cv2

def process(path):
  cap = cv2.VideoCapture(path)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('output8.avi', fourcc, 30,
   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  kernel = np.ones((5, 5), np.uint8)
  fgbg = cv2.createBackgroundSubtractorKNN()
  while cap.isOpened():
    stat, frame = cap.read()
    if not stat:
      break
    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask, kernel, iterations = 1)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    # gradX = cv2.convertScaleAbs(cv2.Sobel(fgmask, cv2.CV_16S, 1, 0))
    # gradY = cv2.convertScaleAbs(cv2.Sobel(fgmask, cv2.CV_16S, 0, 1))
    # edges = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
    edges = cv2.Canny(fgmask, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 30)
    if lines is None:
      out.write(frame)
      continue
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for singleLine in lines[0:10]:
      for rho, theta in singleLine:
        cosine = np.cos(theta)
        sine = np.sin(theta)
        x0 = cosine * rho
        y0 = sine * rho
        x1.append(int(x0 + 400 * (-sine)))
        y1.append(int(y0 + 400 * (cosine)))
        x2.append(int(x0 - 400 * (-sine)))
        y2.append(int(y0 - 400 * (cosine)))
    finalx1 = int(sum(x1) / len(x1))
    finalx2 = int(sum(x2) / len(x2))
    finaly1 = int(sum(y1) / len(y1))
    finaly2 = int(sum(y2) / len(y2))
    cv2.line(frame, (finalx1, finaly1), (finalx2, finaly2), (255, 0, 255), 2)
    out.write(frame)
  out.release()
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  process('Videos/8.mp4')