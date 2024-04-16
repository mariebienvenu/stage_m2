
import cv2
import numpy as np

import app.optical_flow as oflow
from app.ImageProcessing import ImageProcessing


def no_crop(height, width):
    return {'x1':0, 'x2':width, 'y1':0, 'y2':height}



class Video:
   

    def __init__(self, filepath, verbose=0):
        self.filepath = filepath
        
        vid_capture = cv2.VideoCapture(filepath)
        assert vid_capture.isOpened(), "Error opening the video file"
        ret, frame = vid_capture.read()
        assert ret, "Error reading the video file"

        self.fps = vid_capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height, self.frame_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.wait_time = int(1000/self.fps) #in milliseconds
        self.n_channels = frame.shape[2] #cv2.CAP_PROP_CHANNEL and cv2.CAP_PROP_VIDEO_TOTAL_CHANNELS do not work as intended
        self.is_loaded = False

        vid_capture.release()

        if verbose>0:
            print(f'VIDEO DETAILS: \n\t Frame rate: {self.fps} FPS \n\t Frame count: {self.frame_count} \n\t Frame dimension: ({self.frame_height},{self.frame_width}) \n\t Number of color channels: {self.n_channels}')


    def play(self, close_key='q'):
        vid_capture = cv2.VideoCapture(self.filepath)
        while(vid_capture.isOpened()):
            ret, frame = vid_capture.read() # (bool, numpy array of frame)
            if ret == True:
                cv2.imshow('Video',frame)
                key = cv2.waitKey(self.wait_time)
                if key == ord(close_key):
                    break
            else:
                break
        vid_capture.release()
        cv2.destroyAllWindows()


    def load(self, verbose=0):
        if self.is_loaded:
            if verbose>0:
                print('Video already loaded in memory.')
            return self.video_content
        self.video_content = np.zeros((self.frame_count, self.frame_height, self.frame_width, self.n_channels), dtype=np.uint8)
        vid_capture = cv2.VideoCapture(self.filepath)
        index = 0
        while(vid_capture.isOpened()):
            ret, frame = vid_capture.read() # (bool, numpy array of frame)
            if ret == True:
                self.video_content[index, :, :, :] = np.copy(frame)
            else:
                break
            index += 1
        vid_capture.release()
        cv2.destroyAllWindows()
        self.is_loaded = True
        if verbose>0:
            print('Video now loaded in memory.')
        return self.video_content
    

    def play_frame_by_frame(self, close_key='q', next_key='z', previous_key='a', start_frame=None):
        self.load()
        start = start_frame if start_frame is not None else self.frame_count//2
        diff = 0
        while True:
            current = max(0, min(start+diff, self.frame_count))
            cv2.imshow("Frame by frame", self.video_content[current,:,:,:])
            key = cv2.waitKey(0)
            if key == ord(close_key):
                cv2.destroyWindow("Frame by frame")
                break
            elif key == ord(previous_key) and start+diff>=1:
                diff -= 1
            elif key == ord(next_key) and start+diff<self.frame_count-1:
                diff += 1
            print(f'Displaying frame {start+diff}')


    def get_frame(self, index, image_processing : ImageProcessing|str = ImageProcessing.none, crop=None):
        crop = no_crop(self.frame_height, self.frame_width) if crop is None else crop
        self.load()
        assert index>=0 and index<self.frame_count, f"Index {index} out of video range [0, {self.frame_count}]."
        frame = np.copy(self.video_content[index,crop['y1']:crop['y2'],crop['x1']:crop['x2'],:])
        try:
            return image_processing(frame)
        except TypeError: # typically, "String is not callable"
            return getattr(ImageProcessing, image_processing)(frame)


    def get_optical_flow(self, index, image_processing=ImageProcessing.gray, crop=None, degrees=True, **kwargs): # TODO Video.get_optical_flow() -- should not be here ?
        '''if background_proportion is 0 then there will be no thresholding'''
        assert index>=0 and index<self.frame_count-1, f"Index out of video's optical flow range: {index} should be between 0 and {self.frame_count-1} but is not."
        frame1, frame2 = self.get_frame(index, image_processing=image_processing, crop=crop), self.get_frame(index+1, image_processing=image_processing, crop=crop)
        flow = oflow.OpticalFlow.compute_oflow(frame1, frame2, use_degrees=degrees, **kwargs)
        return flow
    

    def get_spatial_crop_input_from_user(self, initial_box : dict = None, verbose=0):
        global frame_idx, x1, x2, y1, y2, drawing, frame
        frame_idx = 0
        close_key, previous_key, next_key = 'q', 'b', 'n' #quit, before, next


        frame = self.get_frame(frame_idx)
        x1, x2, y1, y2 = 0, frame.shape[1], 0, frame.shape[0]
        if initial_box is not None:
            x1, x2, y1, y2 = initial_box.values()
        drawing = False

        def draw_rectangle(event, x, y, flags, param):
            global frame_idx, x1, x2, y1, y2, drawing, frame
            frame = self.get_frame(frame_idx)
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                x1, y1 = x,y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing == True:
                    x2, y2 = x,y
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x2, y2 = x,y

        cv2.namedWindow("Get crop zone from user", flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED) # does not work as intended
        cv2.resizeWindow("Get crop zone from user", 100, 100)
        cv2.setMouseCallback("Get crop zone from user", draw_rectangle)

        while True:
            current = max(0, min(frame_idx, self.frame_count))
            frame = self.get_frame(current)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.imshow("Get crop zone from user", frame)
            key = cv2.waitKey(1)# & 0xFF
            if key == ord(close_key):
                cv2.destroyWindow("Get crop zone from user")
                break
            elif key == ord(previous_key) and frame_idx >= 1:
                frame_idx -= 1
            elif key == ord(next_key) and frame_idx < self.frame_count-1:
                frame_idx += 1
            print(f'Displaying frame {frame_idx}') if verbose>0 else None

        return {'x1':min(x1,x2), 'x2':max(x1,x2), 'y1':min(y1,y2), 'y2':max(y1,y2)}
    
    
    @staticmethod # TODO Video.from_array() -- maybe should be a classmethod since it is a named constructor ? -> classmethods are better if there is inheritance & overload
    def from_array(array : np.ndarray, filepath='/tmp.mp4', fps=30, verbose=0):
        # saves, returns Video object
        frame_count = array.shape[0]
        frame_height, frame_width = array.shape[1], array.shape[2]
        if len(array.shape)<4:
            array = np.expand_dims(array, axis=3)
        n_channels = array.shape[3]
        is_color = n_channels != 1
        fourcc = None
        if '.mp4' in filepath:
            fourcc = 'mp4v'
        elif '.avi' in filepath:
            fourcc = 'XVID'
        assert fourcc is not None, f'Failed to initialize proper video codec for "{filepath}" file.'
        vid_writer = cv2.VideoWriter(
            filepath,
            fourcc=cv2.VideoWriter_fourcc(*fourcc),
            fps=fps,
            frameSize=(frame_width, frame_height),
        )
        
        for i in range(frame_count):
            frame = array[i,:,:,:].astype(np.uint8)
            if not is_color:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            vid_writer.write(frame)
        vid_writer.release()

        return Video(filepath, verbose)
    
