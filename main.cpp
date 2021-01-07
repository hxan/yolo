#include<iostream>
#include<fstream> 
#include<sys/time.h>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core/ocl.hpp>
#include"darknet.h"

using namespace std;
using namespace cv;


class YoloDetect {
    private:

        network *net;
        int classes;
        float thresh;
        float nms;
        vector<string> namesMap;
        
        void imgConvert(float* srcImg,Mat &rgbImg){
            
            int h = rgbImg.rows;
            int w = rgbImg.cols;
            int c = rgbImg.channels();
            
            for(int k= 0; k < c; ++k){
                for(int i = 0; i < w; ++i){
                    for(int j = 0; j < h; ++j){
                        srcImg[k*w*h+i*w+j] = rgbImg.data[(i*w + j)*c + k]/255.;
                    }
                }
            }
        }


    public:
        YoloDetect(char *cfgName,char *weightName,char *names){

            this->classes = 80;
            this->thresh = 0.7;
            this->nms = 0.35;

            this->net = load_network(cfgName,weightName,0 );
            set_batch_network(this->net, 1);

            ifstream classNamesFile(names);

            string className = "";
            if (classNamesFile.is_open()){
                while (getline(classNamesFile, className)){
                    this->namesMap.push_back(className);
                }
            }
        }


        
        void detectFrame(Mat &frame,vector<Rect> &boxes,vector<String> &labels){//

            Mat rgbImg;
            cvtColor(frame, rgbImg, COLOR_BGR2RGB); //frame(Mat)->rgbImg(Mat)->matResizeImg(Mat)->resizeImg(float*)
        
            Mat matResizeImg;
            resize(rgbImg,matResizeImg,Size(this->net->w,this->net->h),INTER_NEAREST);   

            float* resizeImg = (float*)malloc( this->net->w*this->net->h*3*sizeof(float) );
            this->imgConvert(resizeImg,matResizeImg);

            network_predict(this->net,resizeImg);//网络推理

            int nboxes = 0;
            detection *dets = get_network_boxes(this->net,rgbImg.cols,rgbImg.rows,this->thresh,0.5,0,1,&nboxes);

            if(this->nms > 0){
                do_nms_sort(dets,nboxes,this->classes,this->nms);
            }

            boxes.clear();

            vector<int> classNames;

            for (int i = 0; i < nboxes; i++) {

                bool flag = false;
                int className;

                for(int j = 0; j < (this->classes); j++){
                    if( (dets[i].prob[j]>(this->thresh)) &&(!flag) ){
                        flag = true;
                        className = j;
                    }
                }
                if(flag)
                {
                    int left = (dets[i].bbox.x - dets[i].bbox.w / 2.)*frame.cols;
                    int right = (dets[i].bbox.x + dets[i].bbox.w / 2.)*frame.cols;
                    int top = (dets[i].bbox.y - dets[i].bbox.h / 2.)*frame.rows;
                    int bot = (dets[i].bbox.y + dets[i].bbox.h / 2.)*frame.rows;

                    if (left < 0) left = 0;
                    if (right > frame.cols - 1) right = frame.cols - 1;
                    if (top < 0) top = 0;
                    if (bot > frame.rows - 1) bot = frame.rows - 1;

                    Rect box(left, top, fabs(left - right), fabs(top - bot));

                    boxes.push_back(box);
                    
                    String label = String(this->namesMap[className]); 
                    labels.push_back(label);
                }
            }
            free_detections(dets, nboxes);

        }

        ~YoloDetect(){
            free_network(this->net);
        }
};




int main(void)
{
    VideoCapture capture(0);

    namedWindow("video",WINDOW_NORMAL);

    YoloDetect yoloDetect("/home/l/darknet/cfg/yolov3.cfg",
                          "/home/l/darknet/yolov3.weights",
                          "/home/l/darknet/data/coco.names");

    while(true){

        
        vector<Rect> rects;
        vector<String> labels;
        
        Mat frame;
        if ( !capture.read(frame) ){
            printf("fail to read.\n");
            return 0;
        }

        yoloDetect.detectFrame(frame,rects,labels);

        for(int i = 0; i < rects.size(); i++)
        {
            rectangle(frame,rects[i],Scalar(0,0,255),2);
            
            Size labelSize = getTextSize(labels[i],0, 0.5, 1, (int*)0);
            putText(frame,labels[i],Point(rects[i].x,rects[i].y+labelSize.height),0,0.5,Scalar(0, 0, 255),1);
        }

        imshow("video",frame);
        waitKey(1);
    }
}




















