import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import argparse

from tqdm import tqdm


"""!
@brief      Apply filters to the frame

@details    This function apply filters to the frame
            Convert to grayscale
            Apply threshold filter
            Cut ROI
            Resize image
            Apply dilation filter

@param      frame  The frame
@param      roi    The roi

@return     The frame with filters applied
"""
def apply_filters(frame, roi):
    try:
        #Convert to grayscale
        frameF = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Apply threshold filter
        frameF = cv2.threshold(frameF, 50, 255, cv2.THRESH_BINARY_INV)[1]
        # frameF = cv2.adaptiveThreshold(frameF, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 9)
        # frameF = cv2.adaptiveThreshold(frameF, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 7)

        #Cut ROI
        r = roi
        frameF = frameF[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        #resize image
        frameF = cv2.resize(frameF, (0, 0), fx=3, fy=3)
        
        #Apply dilation filter
        frameF = cv2.dilate(frameF, np.ones((7, 7), np.uint8), iterations=1)

        return frameF
    except Exception as e:
        tqdm.write(f"Error in apply_filters: {e}")
    return None

"""!
@brief      Process the ocr result

@details    This function process the ocr result.
            Check if the result is a number.
            Check if the confidence is greater than defined threshold.
            If the result is a number and the confidence is greater than defined threshold.
            the result is added to the results list.
            If the result is a number more than (1+mm_threshold) times or less than mm_threshold times the average of the last 10 values
            the error count is increased, and a polynomial fit is applied to the results list. 
            If the result is not a number or the confidence is less than defined threshold
            the error count is increased, and a polynomial fit is applied to the results list.
            The polynomial fit is applied to the last 10 values of the results without fit flag.
            
@param      orc_result  The orc result
@param      results     The results previously processed (read only)
@param      WFIT        The window fit the polynomial (optional)
@param      confidence_threshold  The confidence threshold (optional)
@param      mm_threshold  The mm threshold (optional)

@return[0] True if the result is processed, False if the result is not processed
@return[1] The result processed, dictionary with the value, confidence and fit flag
"""
def process_orc_result(orc_result, results, WFIT=100, confidence_threshold=0.7, mm_threshold=0.5, poly_degree=1):
    try:
        readed = False
        value = 0
        confidence = 0

        if len(orc_result) > 0:

            #Replace ',' and ';' for '.' the OCR misread the decimal point
            #and convert to float
            value = float(orc_result[0][1].replace(',', '.').replace(';', '.'))
            confidence = float(orc_result[0][2])

            if confidence > confidence_threshold:
                #calculate the average of the last 10 values
                #and if the current value is inside the range of the average
                #accept the current value
                realResults = [x for x in results if x['fit'] == False]
                if len(realResults) > WFIT:
                    result2 = realResults[-WFIT:]
                    avg = sum([x['value'] for x in result2]) / len(result2)
                    readed = True if value > avg * mm_threshold and value < avg * (1+mm_threshold) else False
                else:
                    readed = True if value < 1 and value > 0 else False

        if readed:
            return True, {'value': value, 'confidence': confidence, 'fit': False}

        #if the has an exeception, fit a polynomial with the last 10 values
        realResults = [x for x in results if x['fit'] == False]
        if len(realResults) > WFIT:
            result2 = realResults[-WFIT:]

            x = [x for x in range(len(result2))]
            y = [x['value'] for x in result2]
            value = np.polyval(np.polyfit(x, y, poly_degree), len(result2)+1)

            value = value if value < 10 else realResults[-1]['value']
            value = value if value > 0 else realResults[-1]['value']
            value = np.round(value, 1)

        elif len(realResults) > 0:
            #if the list has less than 10 values, repeat the last value
            value = results[-1]['value']
        else:
            value = 0

        return True, {'value': value, 'confidence': 0, 'fit': True}

    except Exception as e:
        tqdm.write("Failed to process the result: " + str(e))

    return False, {'value': 0, 'confidence': 0, 'fit': True}

"""!
@brief      Draw the result on the frame

@details    This function draw the result on the frame.
            This function insert the processed frame in the raw frame.
            This function insert the plot in the raw frame.
            This function write the result in the raw frame and if the result is fitted, the result is in red.

@param      value       The value
@param      fit         The fit flag
@param      pFrame      The processed frame, Gray scale 
@param      rFrame      The raw frame, BGR color
@param      frame_count The frame count
@param      results     The results
@param      lRead       Axis line to plot the read values
@param      lFit        Axis line to plot the fitted values
@param      ax          The axis of the lRead and lFit of the plot
@param      offsetProcessed  The offset processed (optional)
@param      offsetPlot  The offset plot (optional)

@return     The raw frame with the result in BGR, or None if the result is not processed
"""
def draw_results(value, fit, pFrame, rFrame, frame_count, results, lRead, lFit, ax, offsetProcessed=(680, 650), offsetPlot = (1219, 0)):
    try:
        #Copy the raw frame to draw the results
        dFrame = rFrame.copy()

        #Set the text to print
        strValue = "Read: " + str(value) if not fit else "Read: " + str(value) + " (Fitted)"
        color = (0, 255, 0) if not fit else (0, 0, 255)

        #Convert the processed frame to BGR
        pFrame = cv2.cvtColor(pFrame, cv2.COLOR_GRAY2BGR)
        
        #Convert the offset to int
        offsetProcessed = int(offsetProcessed[0]), int(offsetProcessed[1])
        offsetPlot = int(offsetPlot[0]), int(offsetPlot[1])

        #Insert in the raw_frame the processed frame
        dFrame[offsetProcessed[1]:offsetProcessed[1]+pFrame.shape[0], offsetProcessed[0]:offsetProcessed[0]+pFrame.shape[1]] = pFrame

        cv2.putText(dFrame, strValue, (offsetProcessed[0], int(offsetProcessed[1]+pFrame.shape[0]*1.2)), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 10, cv2.LINE_AA)

        #plot read values as green dots
        lRead.set_data([x['index'] for x in results if x['fit'] == False], [x['value'] for x in results if x['fit'] == False])

        #plot fitted valueues as red dots
        lFit.set_data([x['index'] for x in results if x['fit'] == True], [x['value'] for x in results if x['fit'] == True])

        ax.set_xlim([0, frame_count])
        ax.set_ylim([0, np.max([x['value'] for x in results])*1.2])
        ax.legend(loc=0)

        #transform the plot in an image and insert in the raw_frame
        fig = plt.gcf()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        #Insert in the raw_frame the processed frame,
        dFrame[offsetPlot[1]:offsetPlot[1]+data.shape[0], offsetPlot[0]:offsetPlot[0]+data.shape[1]] = data

        return dFrame
    except Exception as e:
        tqdm.write("Fails to draw the results: " + str(e))
        return None

"""!
@brief      Draw the plot on a file

@details    This function draw the plot on a png file

@param      results     The results
@param      filename    The filename
@param      figsize     The figsize (optional)

"""
def draw_plot(results, filename, figsize=(7, 6)):
    try:
        plt.figure(figsize=figsize)

        #plot read values as green dots
        plt.plot([x['index'] for x in results if x['fit'] == False], [x['value'] for x in results if x['fit'] == False], 'go', label='Read')

        #plot fitted valueues as red dots
        plt.plot([x['index'] for x in results if x['fit'] == True], [x['value'] for x in results if x['fit'] == True], 'ro', label='Fitted')

        plt.xlabel('Frame')
        plt.ylabel('Value')
        plt.legend(loc=0)

        plt.savefig(filename)
    except Exception as e:
        tqdm.write("Fails to draw the plot: " + str(e))

"""!
@brief      Get the roi coordinates

@details    This function get the roi coordinates
            If the roi is not defined, the function ask the user to select the roi
            using opencv selectROI function

@param      roi  The roi to be checked
@param      video to get the images to select the roi

@return     The roi coordinates
"""
def get_roi(roi, video):
    if roi is None or len(roi) != 4:
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        while not ret:
            ret, frame = cap.read()
        roi = cv2.selectROI(frame)
        cap.release()

    if sum(roi) == 0:
        raise Exception("The roi is not defined")

    return roi

"""!
@brief      Main function

@details    This function is the main function of the program

@param      args  The arguments
"""
def run(video, name, draw_frame, roi, WFIT, confidence_threshold, mm_threshold, poly_degree, offsetProcessed, offsetPlot, figSize):

    start_time = time.time()

    #Read video with OpenCV
    cap = cv2.VideoCapture(video)

    # Get roi coordinates
    r = roi

    text_reader = easyocr.Reader(['en'])

    #Get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    fps = fps if fps > 0 else 30

    #Get total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    #Write a video with OpenCV
    out = None

    frame_reads = []
    error_count = 0
    frame_count = 0

    fig, ax = plt.subplots(figsize=figSize)
    lRead, = ax.plot([], [], 'g.', label='Read')
    lFit, = ax.plot([], [], 'r.', label='Fitted')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Value')

    pbar = tqdm(total=total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rawFrame = frame.copy()

        #Apply the filters to the frame
        frame = apply_filters(frame, r)

        #Check if the frame is valid
        if frame is None:
            error_count += 1
            frame_reads.append({'index': frame_count, 'value': -1, 'confidence': -1, 'fit': True})
            continue

        #Get the result from the OCR
        result = text_reader.readtext(frame)

        #Process the result
        result = process_orc_result(result, frame_reads, WFIT, confidence_threshold, mm_threshold, poly_degree)
        
        frame_count += 1
        pbar.update(1)

        #Check if the result is valid
        if not result[0]:
            error_count += 1
            frame_reads.append({'index': frame_count, 'value': -1, 'confidence': -1,'fit': True})
            continue

        #Check if result is a fit increment the error count
        if result[1]['fit']:
            error_count += 1

        #Check if the video writer is not defined and draw_frame is True
        if out is None and draw_frame:
            [h, w] = rawFrame.shape[:2]
            video_name = os.path.join(name, 'output.avi')
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

        #Append the result to the frame_reads list
        frame_reads.append({'index': frame_count, 'value': result[1]['value'], 'confidence': result[1]['confidence'], 'fit': result[1]['fit']})

        #Check if the video writer is defined and draw_frame is True
        if out is not None and draw_frame:
            
            #Draw the result on the frame
            rFrame = draw_results(result[1]['value'], result[1]['fit'], frame, rawFrame, frame_count, frame_reads, lRead, lFit, ax, offsetProcessed, offsetPlot)
            
            #Write the frame on the video
            if rFrame is not None:
                out.write(rFrame)

        #Print the result on the console
        tqdm.write("frame: {} \tvalue: {} \tconfidence: {} \tfit: {} \terror count: {}".format(frame_count, result[1]['value'], result[1]['confidence'], result[1]['fit'], error_count))

    if out is not None:
        out.release()

    cap.release()

    #Draw the plot
    draw_plot(frame_reads, os.path.join(name, 'plot.png'), figsize=figSize)

    #Save the results in a csv file
    df = pd.DataFrame(frame_reads)
    df.to_csv(os.path.join(name, 'results.csv'), index=False)

    tqdm.write("Total time: {}".format(time.time() - start_time))
    pbar.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="path to input video")
    ap.add_argument("-p", "--project", default='experiments',
        help="path to project folder")
    ap.add_argument("-n", "--name", default='runs',
        help="name of the experiment")
    ap.add_argument("-d", "--draw", action='store_true',
        help="draw the results in a video")
    ap.add_argument("-r", "--roi", default=None, nargs='+', type=int,
        help="roi coordinates")
    ap.add_argument("-w", "--wfit", default=100,
        help="size of the window to fit the polynomial correction")
    ap.add_argument("-c", "--confidence", default=0.7,
        help="confidence threshold of the ocr detection to consider a digit")
    ap.add_argument("-m", "--mm", default=0.5,
        help="moving average threshold to consider a digit")
    ap.add_argument("-pd", "--polydegree", default=1,
        help="polynomial degree to fit the correction")
    ap.add_argument("-opim", "--offsetprocessed", default=(680, 650), nargs='+', type=int,
        help="offset of the processed frame to draw inside the original frame, used only if draw is true")
    ap.add_argument("-oplt", "--offsetplot", default=(1219, 0), nargs='+', type=int,
        help="offset of the plot to draw inside the original frame, used only if draw is true")
    ap.add_argument("-fs", "--figsize", default=(7, 6), nargs='+', type=float,
        help="figure size in matplotlib, used only if draw is true")
    
    args = vars(ap.parse_args())

    #Check if the video exists
    if not os.path.exists(args['video']):
        print("Video not found")
        exit()

    #Create the folder project and the name of the experiment
    #If the folder already exists, generete a new name for the experiment
    #with a incremental number
    if not os.path.exists(args['project']):
        os.mkdir(args['project'])

        args['name'] = os.path.join(args['project'], args['name'])
        os.mkdir(args['name'])
    else:
        i = 1
        while os.path.exists(os.path.join(args['project'], args['name'] + str(i))):
            i += 1
        args['name'] = os.path.join(args['project'], args['name'] + str(i))
        os.mkdir(args['name'])

    args['roi'] = get_roi(args['roi'], args['video'])

    run(args['video'], args['name'], args['draw'], args['roi'], args['wfit'], args['confidence'], args['mm'], args['polydegree'], args['offsetprocessed'], args['offsetplot'], args['figsize'])

