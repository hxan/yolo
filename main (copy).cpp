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

inline void imgConvert(float* srcImg,Mat &rgbImg){
    
    uchar *data = rgbImg.data;

    int h = rgbImg.rows;
    int w = rgbImg.cols;
    int c = rgbImg.channels();
    
    for(int k= 0; k < c; ++k){
        for(int i = 0; i < h; ++i){
            for(int j = 0; j < w; ++j){
                srcImg[k*w*h+i*w+j] = data[(i*w + j)*c + k]/255.;
            }
        }
    }
}

void resizeInner(float *src, float* dst,int srcW,int srcH,int dstW,int dstH){

    size_t sizePa = dstW*srcH*3*sizeof(float);
    float* part=(float*)malloc(sizePa);

    float w_scale = (float)(srcW - 1) / (dstW - 1);
    float h_scale = (float)(srcH - 1) / (dstH - 1);

    for(int k = 0; k < 3; ++k){
        for(int r = 0; r < srcH; ++r){
            for(int c = 0; c < dstW; ++c){
                float val = 0;
                if(c == dstW-1 || srcW == 1){
                    val=src[k*srcW*srcH+r*srcW+srcW-1];
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val=(1 - dx) * src[k*srcW*srcH+r*srcW+ix] + dx * src[k*srcW*srcH+r*srcW+ix+1];
                }
                part[k*srcH*dstW + r*dstW + c]=val;
            }
        }
    }

    for(int k = 0; k < 3; ++k){
        for(int r = 0; r < dstH; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(int c = 0; c < dstW; ++c){
                float val = (1-dy) * part[k*dstW*srcH+iy*dstW+c];
                dst[k*dstW*dstH + r*dstW + c]=val;
            }
            if(r == dstH-1 || srcH == 1)
                continue;
            for(int c = 0; c < dstW; ++c){
                float val = dy * part[k*dstW*srcH+(iy+1)*dstW+c];
                dst[k*dstW*dstH + r*dstW + c]+=val;
            }
        }
    }
    free(part);
}

void imgResize(float *src, float* dst,int srcW,int srcH,int dstW,int dstH){

    int new_w = srcW;
    int new_h = srcH;
    if (((float)dstW/srcW) < ((float)dstH/srcH)) {
        new_w = dstW;
        new_h = (srcH * dstW)/srcW;
    } else {
        new_h = dstH;
        new_w = (srcW * dstH)/srcH;
    }

    size_t sizeInner=new_w*new_h*3*sizeof(float);
    float* ImgReInner=(float*)malloc(sizeInner);
    resizeInner(src,ImgReInner,srcW,srcH,new_w,new_h);
    for(int i=0;i<dstW*dstH*3;i++){
        dst[i]=0.5;
    }
    for(int k = 0; k < 3; ++k){
        for(int y = 0; y < new_h; ++y){
            for(int x = 0; x < new_w; ++x){
                float val = ImgReInner[k*new_w*new_h+y*new_w+x];
                dst[k*dstH*dstW + ((dstH-new_h)/2+y)*dstW + (dstW-new_w)/2+x]=val;
            }
        }
    }
    free(ImgReInner);
}

int main()
{
    float thresh = 0.7;//参数设置
    float nms = 0.35;
    int classes = 80;

    VideoCapture capture(0);
    namedWindow("video",WINDOW_NORMAL);

    network *net=load_network( (char *)"/home/l/darknet/cfg/yolov3.cfg", (char *)"/home/l/darknet/yolov3.weights",0 );
    set_batch_network(net, 1);

    

    vector<string> classNamesVec;
    ifstream classNamesFile("/home/l/darknet/data/coco.names");
    if (classNamesFile.is_open()){
        string className = "";
        while (getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }
    

    while(1)
    {
        Mat frame;
        Mat rgbImg;

        if ( !capture.read(frame) ){
            printf("fail to read.\n");
            return 0;
        }

        cvtColor(frame, rgbImg, COLOR_BGR2RGB);

        size_t srcSize = rgbImg.rows*rgbImg.cols*3*sizeof(float);

        float* srcImg = (float*)malloc(srcSize);

        Mat matSrcImg;
        
        resize(rgbImg,matSrcImg,Size(net->w,net->h),INTER_NEAREST);   

        

        imgConvert(srcImg,rgbImg);//将图像转为yolo形式

        size_t resizeSize = net->w*net->h*3*sizeof(float);


        float* resizeImg = (float*)malloc(resizeSize);

        imgResize(srcImg,resizeImg,frame.cols,frame.rows,net->w,net->h);//缩放图像

        network_predict(net,resizeImg);//网络推理

        int nboxes = 0;

        detection *dets = get_network_boxes(net,rgbImg.cols,rgbImg.rows,thresh,0.5,0,1,&nboxes);

        if(nms){
            do_nms_sort(dets,nboxes,classes,nms);
        }

        vector<Rect> boxes;

        boxes.clear();

        vector<int>classNames;

        for (int i = 0; i < nboxes; i++) {
            bool flag=0;
            int className;
            for(int j=0;j<classes;j++){
                if(dets[i].prob[j]>thresh){
                    if(!flag){
                        flag=1;
                        className=j;
                    }
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
                classNames.push_back(className);
            }
        }
        free_detections(dets, nboxes);

        for(int i=0;i<boxes.size();i++)
        {
            rectangle(frame,boxes[i],Scalar(0,0,255),2);
            String label = String(classNamesVec[classNames[i]]);
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            putText(
                    frame, label,
                    Point( boxes[i].x, boxes[i].y+labelSize.height ),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255),
                    1
            );

        }

        imshow("video",frame);
        waitKey(1);

        free(srcImg);
        free(resizeImg);


    }

    free_network(net);
    
    capture.release();
    return 0;
}























