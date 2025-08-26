import sys
import time

from .utils import (
    open_source, load_model, warmup, get_frame, preprocess, infer, postprocess,
    draw_and_compose, maybe_open_writer, write_and_show, cleanup
)




# ========================================== Run pipeline ============================================

# calling this function will run all pipelines
def main():
    # ---------------------- input / output settings -------------

    # src = sys.argv[1] if len(sys.argv) > 1 else 0
    src = sys.argv[1] if len(sys.argv) > 1 else 0  # use first argument if given, else webcam
    save_flag = True            # save video
    out_path = 'output.mp4'    # path of output video


    # ----------------------- preparation -----------------------------

    cap = open_source(src)     # open the video source
    model = load_model()       # load YOLOv5
    warmup(model)               # warm up the model to reduce initial lag
    names = model.names        # all class names the model can detect
    writer = None              # initialize writer if needed later
    t_prev = time.time()         # store current system time to calculate FPS


    # ----------------------------------- main loop inside try -------------------------------
    try:
        while True:
            frame = get_frame(cap)               # read frame from video source
            if frame is None:
                break                            # end of video or error

            img = preprocess(frame)               # not needed for YOLOv5 (already handles it)

            result = infer(model, img, imgsz=640)    # run inference

            # post process: get class labels and count them
            labels, counts = postprocess(result, names)

            # ----------------------- FPS calculation ------------
            now = time.time()
            dt = max(now - t_prev, 1e-6)
            fps = 1.0 / dt
            t_prev = now

            # draw output and add header on top of frame
            vis = draw_and_compose(result, counts, fps)

            # create VideoWriter if saving is enabled
            writer = maybe_open_writer(save_flag, writer, vis, out_path=out_path, fps=30)

            # --------------------- Save frame and show -------------------
            # save frame to file if writer exists and show on screen
            write_and_show(writer, vis)

            # exit the program by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally: 
        # -------------- Clean up all resources -------------
        cleanup(cap, writer)


# Correct way to check main entry point
if __name__ == "__main__":
    main()
