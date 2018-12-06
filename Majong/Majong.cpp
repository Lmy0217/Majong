// Majong.cpp
//

#include "stdafx.h"
#include "Majong.h"


void callBack(int, void *userdata) {

    color_callback_data ccb = SHOW_CALLBACK ? (*(color_callback_data*)(userdata)).clone()
        : (*(color_callback_data*)(userdata));

    Mat mask;
    inRange(ccb.hsv, Scalar(ccb.hmin, ccb.smin / 255.0, ccb.vmin / 255.0),
        Scalar(ccb.hmax, ccb.smax / 255.0, ccb.vmax / 255.0), mask);

    for (int r = 0; r < ccb.bgr.rows; r++) {
        for (int c = 0; c < ccb.bgr.cols; c++) {
            if (mask.at<uchar>(r, c) != 255) {
                if (ccb.nomask != -1) {
                    ccb.dst.at<Vec3b>(r, c) = Vec3b(ccb.nomask, ccb.nomask, ccb.nomask);
                }
            }
            else {
                if (ccb.mask != -1) {
                    ccb.dst.at<Vec3b>(r, c) = Vec3b(ccb.mask, ccb.mask, ccb.mask);
                }
            }
        }
    }
    if (SHOW_CALLBACK) {
        imshow(ccb.dstName, ccb.dst);
    }

    ccb.dst.convertTo(ccb.dst, CV_8UC3, 255.0, 0);
}


void Recognition::color(Mat &img, Mat &dst, color_callback_data &ccb) {

    Mat bgr;
    img.convertTo(bgr, CV_32FC3, 1.0 / 255, 0);

    Mat hsv;
    cvtColor(bgr, hsv, COLOR_BGR2HSV);

    ccb.img = img;
    ccb.bgr = bgr;
    ccb.hsv = hsv;
    ccb.dst = dst;

    if (SHOW_CALLBACK) {
        namedWindow(ccb.dstName, WINDOW_GUI_EXPANDED);
        createTrackbar("hmin", ccb.dstName, &ccb.hmin, 360, callBack, &ccb);
        createTrackbar("hmax", ccb.dstName, &ccb.hmax, 360, callBack, &ccb);
        createTrackbar("smin", ccb.dstName, &ccb.smin, 255, callBack, &ccb);
        createTrackbar("smax", ccb.dstName, &ccb.smax, 255, callBack, &ccb);
        createTrackbar("vmin", ccb.dstName, &ccb.vmin, 255, callBack, &ccb);
        createTrackbar("vmax", ccb.dstName, &ccb.vmax, 255, callBack, &ccb);
    }

    callBack(0, &ccb);
}


double Recognition::getOverlap(const Rect &box1, const Rect &box2) {

    if ((box1.x > box2.x + box2.width) || (box1.y > box2.y + box2.height)
        || (box1.x + box1.width < box2.x) || (box1.y + box1.height < box2.y)) {
        return 0;
    }
    double colInt = abs(min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x));
    double rowInt = abs(min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y));
    double overlapArea = colInt * rowInt;
    double area1 = box1.width * box1.height;
    double area2 = box2.width * box2.height;
    double area1Ratio = overlapArea / area1;
    double area2Ratio = overlapArea / area2;

    if (area1 < area2 && area1Ratio >= 0.5) {
        return -1;
    }
    if (area1 > area2 && area2Ratio >= 0.5) {
        return 1;
    }

    return 0;
}


vector<Point> Recognition::getEdgePoint(vector<Point> &contour) {

    Point leftTop(INT_MAX, INT_MAX), rightTop(INT_MIN, INT_MAX), leftBottom(INT_MAX, INT_MIN), rightBottom(INT_MIN, INT_MIN);

    for (int i = 0; i < contour.size(); i++) {
        if (contour.at(i).x + contour.at(i).y < (double)leftTop.x + leftTop.y) {
            leftTop.x = contour.at(i).x;
            leftTop.y = contour.at(i).y;
        }
        if (contour.at(i).y - contour.at(i).x < (double)rightTop.y - rightTop.x) {
            rightTop.x = contour.at(i).x;
            rightTop.y = contour.at(i).y;
        }
        if (contour.at(i).x - contour.at(i).y < (double)leftBottom.x - leftBottom.y) {
            leftBottom.x = contour.at(i).x;
            leftBottom.y = contour.at(i).y;
        }
        if (contour.at(i).x + contour.at(i).y > (double)rightBottom.x + rightBottom.y) {
            rightBottom.x = contour.at(i).x;
            rightBottom.y = contour.at(i).y;
        }
    }

    return vector<Point>{ leftTop, rightTop, leftBottom, rightBottom };
}


vector<Vec<double, 5>> Recognition::find(Mat &img, int areaThreshold, int minSide) {

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point> approx;
    vector<Rect> rects;
    Rect rect;
    vector<Vec<double, 5>> paralls;
    Vec<double, 5> parall;

    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++) {
        if (fabs(contourArea(contours.at(i))) > areaThreshold) {
            rect = boundingRect(contours.at(i));
            vector<Point> edgePoint = getEdgePoint(contours.at(i));
            double widthDelta = abs(edgePoint.at(1).x - edgePoint.at(0).x) / (double)abs(edgePoint.at(3).x - edgePoint.at(2).x);
            double delta = ((edgePoint.at(2).x - edgePoint.at(0).x) + (edgePoint.at(3).x - edgePoint.at(1).x)) / 2.0;
            if (widthDelta < 0.97 || widthDelta > 1.03) {
                delta = (abs(edgePoint.at(2).x - edgePoint.at(0).x) < abs(edgePoint.at(3).x - edgePoint.at(1).x))
                    ? (edgePoint.at(2).x - edgePoint.at(0).x) : (edgePoint.at(3).x - edgePoint.at(1).x);
            }
            Vec<double, 5> parall(rect.x - min(delta, 0.0), rect.y, rect.width - abs(delta), rect.height, delta);
            if (parall(2) > minSide && parall(3) > minSide) {
                if (rects.size() == 0) {
                    rects.push_back(rect);
                    paralls.push_back(parall);
                }
                else {
                    bool overlapFlag = true;
                    for (int j = (int)rects.size() - 1; j >= 0; j--) {
                        double overlap = getOverlap(rects.at(j), rect);
                        if (overlap == -1) {
                            rects.erase(rects.begin() + j);
                            paralls.erase(paralls.begin() + j);
                            break;
                        }
                        else if (overlap == 1) {
                            overlapFlag = false;
                            break;
                        }
                    }
                    if (overlapFlag) {
                        rects.push_back(rect);
                        paralls.push_back(parall);
                    }
                }
            }
        }
    }

    return paralls;
}


MeanStd Recognition::ms(Mat &img, Vec<double, 5> &pa, int type) {

    MeanStd meanstd;
    meanstd.mean = new vector<double>();
    meanstd.std = new vector<double>();
    Mat tmp_m, tmp_s;
    if (type == 0) {
        for (int i = (int)max(pa(0), pa(0) + pa(4)); i < (int)min(pa(0) + pa(2), pa(0) + pa(2) + pa(4)); i++) {
            meanStdDev(img(Rect(i, 0, 1, (int)pa(3))), tmp_m, tmp_s);
            meanstd.mean->push_back(tmp_m.at<double>(0, 0));
            meanstd.std->push_back(tmp_s.at<double>(0, 0));
        }
    }
    else {
        for (int i = 0; i < pa(3); i++) {
            meanStdDev(img(Rect(int(pa(4) / pa(3) * i - (pa(4) < 0 ? pa(4) : 0)), i, (int)pa(2), 1)), tmp_m, tmp_s);
            meanstd.mean->push_back(tmp_m.at<double>(0, 0));
            meanstd.std->push_back(tmp_s.at<double>(0, 0));
        }
    }

    return meanstd;
}

void Recognition::filter(MeanStd &meanstd, double maxstd) {
    for (int i = 0; i < meanstd.std->size(); i++) {
        if (meanstd.std->at(i) <= maxstd) {
            meanstd.mean->at(i) = 255;
        }
        else {
            meanstd.mean->at(i) = 0;
        }
    }
}

vector<int> Recognition::selectlines(MeanStd &meanstd) {

    vector<int> index;
    int start = -1;
    int flag = 0;

    for (int i = 0; i < meanstd.mean->size(); i++) {
        if (flag == 0 && meanstd.mean->at(i) >= 128.0) {
            flag = 1;
            start = i;
            continue;
        }
        if (flag == 1 && meanstd.mean->at(i) < 128.0) {
            flag = 0;
            index.push_back(int((i - start) / 2.0) + start);
        }
    }

    if (index.size() == 0) {
        index.push_back((int)meanstd.mean->size());
    }

    return index;
}


vector<Vec<double, 5>> Recognition::morlines(Mat &img, Vec<double, 5> &parall, int fullType, int height, int heightDelta,
    double maxstd, double min_ratio, double min_ratio_delta, double min_ratio_delta_type0, double max_ratio,
    double max_ratio_delta_type0) {

    vector<Vec<double, 5>> parall_lines;

    int type = (fullType == 0 || fullType == 1 || fullType == 4 || fullType == 5) ? 0 : 1;
    double rows = round((type == 0 ? sqrt(parall(3) * parall(3) + parall(4) * parall(4)) : parall(2)) / ((fullType == 1
        || fullType == 4 || fullType == 5) ? (double)heightDelta : (double)height));
    rows = rows > 0 ? rows : 1;
    for (int rr = 0; rr < rows; rr++) {
        Vec<double, 5> pa = Vec<double, 5>(type == 0 ? (parall(4) / rows * rr - (parall(4) < 0 ? parall(4) : 0))
            : (parall(2) / rows * rr - (parall(4) < 0 ? parall(4) : 0)),
            type == 0 ? (parall(3) / rows * rr) : 0, type == 0 ? parall(2) : (parall(2) / rows),
            type == 0 ? (parall(3) / rows) : parall(3), type == 0 ? (parall(4) / rows) : parall(4));
        MeanStd meanstd = ms(img, pa, type);
        filter(meanstd, maxstd);
        vector<int> index = selectlines(meanstd);

        double prior = type == 0 ? pa(0) : pa(1);
        int reIndex = -1;
        int reValue = -1;
        for (int i = 0; i < index.size(); i++) {
            double r = double(index.at(i) - prior) / (type == 0 ? sqrt(pa(3) * pa(3) + pa(4) * pa(4)) : pa(2));
            if (r >= (min_ratio - (type == 0 ? ((fullType == 1 || fullType == 4 || fullType == 5)
                ? min_ratio_delta_type0 : 0) : min_ratio_delta)) &&
                r <= (max_ratio - ((fullType == 1 || fullType == 4 || fullType == 5) ? max_ratio_delta_type0 : 0)) &&
                (double((type == 0 ? pa(2) : sqrt(pa(3) * pa(3) + pa(4) * pa(4))) - index.at(i)) / (type == 0
                    ? sqrt(pa(3) * pa(3) + pa(4) * pa(4)) : pa(2)) >= (min_ratio - (type == 0 ? ((fullType == 1
                        || fullType == 4 || fullType == 5) ? min_ratio_delta_type0 : 0) : min_ratio_delta)))) {
                parall_lines.push_back(Vec<double, 5>(parall(0) + (type == 0 ? (parall(4) / rows * rr + prior - pa(0))
                    : (parall(4) / parall(3) * prior + parall(2) / rows * rr)), parall(1) + (type == 0 ? (parall(3) / rows * rr) : prior),
                    type == 0 ? (index.at(i) - prior) : (parall(2) / rows), type == 0 ? (parall(3) / rows) : (index.at(i) - prior),
                    type == 0 ? (parall(4) / rows) : (parall(4) / parall(3) * (index.at(i) - prior))));
                prior = index.at(i);
            }
            else if (r > (max_ratio - ((fullType == 1 || fullType == 4 || fullType == 5) ? max_ratio_delta_type0 : 0))) {
                index.at(i) = int(round((type == 0 ? sqrt(pa(3) * pa(3) + pa(4) * pa(4)) : pa(2)) / ((1 / (min_ratio -
                    (type == 0 ? ((fullType == 1 || fullType == 4 || fullType == 5) ? min_ratio_delta_type0 : 0) : min_ratio_delta))
                    + 1 / (max_ratio - ((fullType == 1 || fullType == 4 || fullType == 5) ? max_ratio_delta_type0 : 0))) / 2) + prior));
                if (i == reIndex && index.at(i) == reValue) {
                    continue;
                }
                reIndex = i;
                reValue = index.at(i);
                i--;
            }
        }
        parall_lines.push_back(Vec<double, 5>(parall(0) + (type == 0 ? (parall(4) / rows * rr + prior - pa(0))
            : (parall(4) / parall(3) * prior + parall(2) / rows * rr)), parall(1) + (type == 0 ? (parall(3) / rows * rr) : prior),
            type == 0 ? (pa(2) + pa(0) - prior) : (parall(2) / rows), type == 0 ? (parall(3) / rows) : (pa(3) + pa(1) - prior),
            type == 0 ? (parall(4) / rows) : (parall(4) / parall(3) * (pa(3) + pa(1) - prior))));
    }

    return parall_lines;
}


bool Recognition::isInParallelogram(Vec2d &point, Vec<double, 5> &parallelogram) {

    return point(1) >= parallelogram(1) && point(1) <= parallelogram(1) + parallelogram(3)
        && parallelogram(3) * (point(0) - parallelogram(0)) >= parallelogram(4) * (point(1) - parallelogram(1))
        && parallelogram(3) * (point(0) - parallelogram(0) - parallelogram(2))
        <= parallelogram(4) * (point(1) - parallelogram(1));
}


int Recognition::range(Rect rect, Mat &img, vector<Vec<double, 5>> &posRect) {

    Vec2d point((rect.x + rect.width / 2) / (double)img.cols, (rect.y + rect.height / 2) / (double)img.rows);

    for (int i = 0; i < posRect.size(); i++) {
        if (isInParallelogram(point, posRect.at(i))) {
            return i;
        }
    }

    return -1;
}


void Recognition::rotate(Mat &img_rect, int type, bool isReversal = true) {

    switch (type) {
    case 0:case 1:
        break;
    case 2:case 3:
        transpose(img_rect, img_rect);
        flip(img_rect, img_rect, 0);
        break;
    case 4:case 5:
        if (isReversal) {
            flip(img_rect, img_rect, 0);
            flip(img_rect, img_rect, 1);
        }
        break;
    case 6:case 7:
        transpose(img_rect, img_rect);
        flip(img_rect, img_rect, 1);
        break;
    default:
        break;
    }
}


vector<Vec<double, 5>> Recognition::getRects(Mat img, color_callback_data &color_callback_data1,
    bool hasErode, int areaThreshold, double minSideRatio) {

    color(img, img, color_callback_data1);
    cvtColor(img, img, COLOR_BGR2GRAY);

    Mat element(2, 2, CV_8U, Scalar(1));
    morphologyEx(img, img, MORPH_OPEN, element);
    if (hasErode) {
        erode(img, img, element);
    }

    if (SHOW_RECTS) {
        namedWindow(color_callback_data1.dstName + "_rects", WINDOW_NORMAL);
        imshow(color_callback_data1.dstName + "_rects", img);
    }

    vector<Vec<double, 5>> paralls = find(img, areaThreshold, int(img.cols * minSideRatio));

    return paralls;
}


Rect Recognition::parall2Rect(Vec<double, 5> &parall, Mat &img) {

    Rect rect((int)min(parall(0), parall(0) + parall(4)), (int)parall(1),
        int(parall(2) + abs(parall(4))), (int)parall(3));
    if (rect.x + rect.width > img.cols) {
        rect.width = img.cols - rect.x;
    }
    if (rect.y + rect.height > img.rows) {
        rect.height = img.rows - rect.y;
    }
    return rect;
}


double Recognition::getOverlap(Vec<double, 5> &para1, Vec<double, 5> &para2, Mat &img) {

    return getOverlap(parall2Rect(para1, img), parall2Rect(para2, img));
}


void Recognition::getOtherRects(vector<Vec<double, 5>> &paralls, vector<Vec<double, 5>> posRect, Mat img,
    color_callback_data &color_callback_data1_othercolor, bool hasErode, int areaThreshold, double minSideRatio) {

    if (color_callback_data1_othercolor.dstName != "no") {
        vector<Vec<double, 5>> paralls_othercolor = getRects(img, color_callback_data1_othercolor,
            hasErode, areaThreshold, minSideRatio);
        for (int i = 0; i < paralls_othercolor.size(); i++) {
            bool overlapFlag = true;
            for (int j = (int)paralls.size() - 1; j >= 0; j--) {
                double overlap = getOverlap(paralls.at(j), paralls_othercolor.at(i), img);
                if (overlap != 0) {
                    overlapFlag = false;
                    break;
                }
            }
            if (overlapFlag) {
                paralls.push_back(paralls_othercolor.at(i));
            }
        }
    }
}


void Recognition::distHandCards(vector<Vec<double, 5>> &parall_small, vector<int> &types, int handCardsType, int minLimit, int maxLimit) {

    for (int i = 0; i < parall_small.size(); i++) {
        if (types.at(i) == 0) {
            double width = parall_small.at(i)(2);
            double height = sqrt(parall_small.at(i)(3) * parall_small.at(i)(3)
                + parall_small.at(i)(4) * parall_small.at(i)(4));
            if (min(width, height) > minLimit && max(width, height) < maxLimit) {
                types.at(i) = handCardsType;
            }
        }
    }
}


void Recognition::getSmall(vector<Vec<double, 5>> &parall_small, vector<int> &types, vector<Vec<double, 5>> &paralls,
    Mat &img, vector<Vec<double, 5>> &posRect, double heightRatio, double heightRatioDelta, double maxstd,
    double min_ratio, double min_ratio_delta, double min_ratio_delta_type0, double max_ratio, double max_ratio_delta_type0,
    double otherLineMinRatio, double otherLineMaxRatio) {

#ifdef SMALL_INDEX
    for (int i = SMALL_INDEX; i <= SMALL_INDEX; i++) {
#else
    for (int i = 0; i < paralls.size(); i++) {
#endif
        Rect rect = parall2Rect(paralls.at(i), img);

        Mat img_rect = img(rect);
        cvtColor(img_rect, img_rect, COLOR_BGR2GRAY);

        int type = range(rect, img, posRect);
        if (type == -1
#ifdef SMALL_TYPE
            || type != SMALL_TYPE
#endif
            ) {
            continue;
        }
        vector<Vec<double, 5>> temp = morlines(img_rect, paralls.at(i), type, (int)(heightRatio * img.cols),
            (int)((heightRatio - heightRatioDelta) * img.cols), maxstd, min_ratio, min_ratio_delta,
            min_ratio_delta_type0, max_ratio, max_ratio_delta_type0);

        if (temp.size() == 0) {
            parall_small.push_back(paralls.at(i));
            types.push_back(type);
        }
        else {
            parall_small.insert(parall_small.end(), temp.begin(), temp.end());
            for (int j = 0; j < temp.size(); j++) {
                types.push_back(type);
            }
        }
    }
    distHandCards(parall_small, types, int(posRect.size()), int(otherLineMinRatio  * img.cols), int(otherLineMaxRatio * img.cols));
}


void Recognition::addDora(vector<Vec<double, 5>> &parall_small, vector<int> &types, MatSize size, vector<Vec<double, 5>> &doraRect, int doraType) {

    for (int i = 0; i < doraRect.size(); i++) {
        parall_small.push_back(Vec<double, 5>(doraRect.at(i)(0) * size[1], doraRect.at(i)(1) * size[0],
            doraRect.at(i)(2) * size[1], doraRect.at(i)(3) * size[0], doraRect.at(i)(4) * size[1]));
        types.push_back(doraType);
    }
}


void Recognition::signSmall(Mat &img, vector<Vec<double, 5>> &parall_small, vector<string> &match_result,
    vector<int> &types, int handCardsType) {

    for (int i = 0; i < parall_small.size(); i++) {
        line(img, Point(int(parall_small.at(i)(0)), int(parall_small.at(i)(1))), Point(int((parall_small.at(i)(0)
            + parall_small.at(i)(2))), int(parall_small.at(i)(1))), Scalar(0, 0, 255), 2);
        line(img, Point(int((parall_small.at(i)(0) + parall_small.at(i)(2))), int(parall_small.at(i)(1))),
            Point(int((parall_small.at(i)(0) + parall_small.at(i)(2) + parall_small.at(i)(4))),
                int((parall_small.at(i)(1) + parall_small.at(i)(3)))), Scalar(0, 0, 255), 2);
        line(img, Point(int((parall_small.at(i)(0) + parall_small.at(i)(2) + parall_small.at(i)(4))),
            int((parall_small.at(i)(1) + parall_small.at(i)(3)))), Point(int((parall_small.at(i)(0)
                + parall_small.at(i)(4))), int((parall_small.at(i)(1) + parall_small.at(i)(3)))), Scalar(0, 0, 255), 2);
        line(img, Point(int((parall_small.at(i)(0) + parall_small.at(i)(4))), int((parall_small.at(i)(1)
            + parall_small.at(i)(3)))), Point(int(parall_small.at(i)(0)), int(parall_small.at(i)(1))), Scalar(0, 0, 255), 2);
        char str[4];
        _itoa_s(i, str, 10);
        putText(img, match_result.at(i), Point((int)parall_small.at(i)(0), (int)(parall_small.at(i)(1) + parall_small.at(i)(3)
            / 2 - 2)), FONT_HERSHEY_DUPLEX, 0.5, (types.at(i) != handCardsType) ? Scalar(255, 0, 0) : Scalar(255, 255, 0), 1);
        putText(img, str, Point((int)parall_small.at(i)(0), (int)(parall_small.at(i)(1) + parall_small.at(i)(3) / 2 + 12)),
            FONT_HERSHEY_DUPLEX, 0.5, (types.at(i) != handCardsType) ? Scalar(255, 0, 0) : Scalar(255, 255, 0), 1);
    }
}


void Recognition::signArea(Mat &img, vector<Vec<double, 5>> &posRect) {

    for (int i = 0; i < posRect.size(); i++) {
        line(img, Point(int(img.cols * posRect.at(i)(0)), int(img.rows * posRect.at(i)(1))), Point(int(img.cols
            * (posRect.at(i)(0) + posRect.at(i)(2))), int(img.rows * posRect.at(i)(1))), Scalar(255, 255, 255), 2);
        line(img, Point(int(img.cols * (posRect.at(i)(0) + posRect.at(i)(2))), int(img.rows * posRect.at(i)(1))),
            Point(int(img.cols * (posRect.at(i)(0) + posRect.at(i)(2) + posRect.at(i)(4))), int(img.rows
                * (posRect.at(i)(1) + posRect.at(i)(3)))), Scalar(255, 255, 255), 2);
        line(img, Point(int(img.cols * (posRect.at(i)(0) + posRect.at(i)(2) + posRect.at(i)(4))), int(img.rows
            * (posRect.at(i)(1) + posRect.at(i)(3)))), Point(int(img.cols * (posRect.at(i)(0) + posRect.at(i)(4))),
                int(img.rows * (posRect.at(i)(1) + posRect.at(i)(3)))), Scalar(255, 255, 255), 2);
        line(img, Point(int(img.cols * (posRect.at(i)(0) + posRect.at(i)(4))), int(img.rows * (posRect.at(i)(1)
            + posRect.at(i)(3)))), Point(int(img.cols * posRect.at(i)(0)), int(img.rows * posRect.at(i)(1))),
            Scalar(255, 255, 255), 2);
    }
}


Platform Recognition::getPlatform(string platName) {

    double resizeRatio;

    color_callback_data color_callback_data1;
    color_callback_data color_callback_data1_othercolor;
    bool hasErode;
    int areaThreshold;
    double otherAreaThresholdRatio;
    double minSideRatio;

    vector<Vec<double, 5>> posRect;
    vector<Vec<double, 5>> doraRect;
    double heightRatio;
    double heightRatioDelta;
    double lineMaxStd;
    double lineMinRatio;
    double lineMinRatioDelta;
    double lineMinRatioDeltaType0;
    double lineMaxRatio;
    double lineMaxRatioDeltaType0;
    double otherLineMinRatio;
    double otherLineMaxRatio;

    bool isReversal;
    vector<string> templetNames = LABEL_NAMES;

    PARMS_PLAT(platName)

    return Platform(resizeRatio, color_callback_data1, color_callback_data1_othercolor, hasErode, areaThreshold,
        otherAreaThresholdRatio, minSideRatio, posRect, doraRect, heightRatio, heightRatioDelta, lineMaxStd,
        lineMinRatio, lineMinRatioDelta, lineMinRatioDeltaType0, lineMaxRatio, lineMaxRatioDeltaType0, otherLineMinRatio,
        otherLineMaxRatio, isReversal, templetNames);
}


map<string, int> Recognition::getVaildPlatforms() {

    return VAILD_PLATFORMS;
}


vector<string> Recognition::DNNMatch(Mat img, vector<Vec<double, 5>> &paralls, vector<int> &types, bool isReversal,
    string model_file, vector<string> &templetNames) {

    vector<string> match_result;

    Net net = readNetFromTensorflow(model_file);

    for (int i = 0; i < paralls.size(); i++) {
        Mat rect = img(parall2Rect(paralls.at(i), img)).clone();
        rotate(rect, types.at(i), isReversal);
        Mat blob = blobFromImage(rect, MODEL_SCALE, Size(MODEL_INPWIDTH, MODEL_INPHEIGHT), Scalar(), true, false);

        net.setInput(blob);
        Mat prob = net.forward();

        Point classIdPoint;
        double confidence;
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;

        match_result.push_back(templetNames.at(classId));
    }

    return match_result;
}


vector<vector<string>> Recognition::getFinallyInfo(vector<string> match_result, vector<Vec<double, 5>> parall_small,
    vector<int> types, Size size, int infoCount) {

    vector<vector<string>> finally_info(infoCount);

    for (int i = 0; i < match_result.size(); i++) {
        char str[256];
        sprintf_s(str, "_%lg_%lg_%lg_%lg_%lg", parall_small.at(i)(0), parall_small.at(i)(1),
            parall_small.at(i)(2), parall_small.at(i)(3), parall_small.at(i)(4));
        finally_info.at(types.at(i)).push_back(match_result.at(i) + str);
    }
    char width_str[12], height_str[12];
    sprintf_s(width_str, "%d", size.width);
    sprintf_s(height_str, "%d", size.height);
    finally_info.push_back(vector<string>{ width_str, height_str });

    return finally_info;
}


void Recognition::createDNNDataset(Mat &img, vector<Vec<double, 5>> &paralls, string dest_filename,
    vector<int> &types, bool isReversal) {

    size_t index = dest_filename.find('.');

    for (int i = 0; i < paralls.size(); i++) {
        Mat img_rect = img(parall2Rect(paralls.at(i), img));
        rotate(img_rect, types.at(i), isReversal);
        char c[10];
        sprintf_s(c, "%03d", i);
        imwrite(dest_filename.substr(0, index) + "_" + c + ".png", img_rect);
    }
}


vector<vector<string>> Recognition::recognize(Instance instance, string dest_filename = "",
    FILE *fMatchWrite = NULL, FILE *fInfoWrite = NULL, string dataset_filename = "") {

    srand((unsigned int)time(NULL));

    clock_t startTime, endTime;

    if (FLAG_CLOCK) {
        startTime = clock();
    }
    Platform platform = getPlatform(instance.platform);
    Mat img = instance.image;
    if (img.data == NULL) {
        printf_s("Error: Invaild image!\n");
        return vector<vector<string>>{ vector<string>{ "Error: Invaild image!" } };
    }
    resize(img, img, Size(int(img.cols * platform.resizeRatio), int(img.rows * platform.resizeRatio)));
    Mat imgclone = img.clone();
    if (FLAG_CLOCK) {
        endTime = clock();
        printf_s("Load Image: ");
        printf_s("%fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
    }

    if (FLAG_CLOCK) {
        startTime = clock();
    }
    vector<Vec<double, 5>> paralls = getRects(img.clone(), platform.color_callback_data1,
        platform.hasErode, platform.areaThreshold, platform.minSideRatio);
    getOtherRects(paralls, platform.posRect, img.clone(), platform.color_callback_data1_othercolor,
        platform.hasErode, int(platform.areaThreshold * platform.otherAreaThresholdRatio), platform.minSideRatio);
    if (FLAG_CLOCK) {
        endTime = clock();
        printf_s("getRects: ");
        printf_s("%fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
    }

    if (FLAG_CLOCK) {
        startTime = clock();
    }
    vector<Vec<double, 5>> parall_small;
    vector<int> types;
    getSmall(parall_small, types, paralls, img, platform.posRect, platform.heightRatio, platform.heightRatioDelta,
        platform.lineMaxStd, platform.lineMinRatio, platform.lineMinRatioDelta, platform.lineMinRatioDeltaType0,
        platform.lineMaxRatio, platform.lineMaxRatioDeltaType0, platform.otherLineMinRatio, platform.otherLineMaxRatio);
    if (parall_small.size() < MIN_CARDLIMIT) {
        printf_s("Error: Less than the minimum detection limit!\n");
        return vector<vector<string>>{ vector<string>{ "Error: Less than the minimum detection limit!" } };
    }
    addDora(parall_small, types, img.size, platform.doraRect, int(platform.posRect.size() + 1));
    if (dataset_filename != "") {
        createDNNDataset(img, parall_small, dataset_filename, types, platform.isReversal);
    }
    if (FLAG_CLOCK) {
        endTime = clock();
        printf_s("getSmall: ");
        printf_s("%fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
    }

    if (FLAG_CLOCK) {
        startTime = clock();
    }
    vector<string> match_result = DNNMatch(img.clone(), parall_small, types, platform.isReversal, MODEL_FILE, platform.templetNames);
    if (fMatchWrite != NULL) {
        fprintf(fMatchWrite, "{\n  \"%s\":{\n", instance.filename.c_str());
        for (int i = 0; i < match_result.size(); i++) {
            if (i < match_result.size() - 1) {
                fprintf(fMatchWrite, "    \"%d\":\"%s\",\n", i, match_result.at(i).c_str());
            }
            else {
                fprintf(fMatchWrite, "    \"%d\":\"%s\"\n", i, match_result.at(i).c_str());
            }
        }
        fprintf(fMatchWrite, "  }\n}\n");
    }
    vector<vector<string>> finallyInfo = getFinallyInfo(match_result, parall_small, types, img.size(), int(platform.posRect.size() + 2));
    if (fInfoWrite != NULL) {
        fprintf(fInfoWrite, "{\n  \"%s\":{\n", instance.filename.c_str());
        for (int i = 0; i < finallyInfo.size(); i++) {
            fprintf(fInfoWrite, "    \"%d\":[\n", i);
            for (int j = 0; j < finallyInfo.at(i).size(); j++) {
                if (j < finallyInfo.at(i).size() - 1) {
                    fprintf(fInfoWrite, "      \"%s\",\n", finallyInfo.at(i).at(j).c_str());
                }
                else {
                    fprintf(fInfoWrite, "      \"%s\"\n", finallyInfo.at(i).at(j).c_str());
                }
            }
            if (i < finallyInfo.size() - 1) {
                fprintf(fInfoWrite, "    ],\n");
            }
            else {
                fprintf(fInfoWrite, "    ]\n");
            }
        }
        fprintf(fInfoWrite, "  }\n}\n");
    }
    if (FLAG_CLOCK) {
        endTime = clock();
        printf_s("getFinallyInfo: ");
        printf_s("%fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
    }

    if (dest_filename != "") {
        if (FLAG_CLOCK) {
            startTime = clock();
        }
        if (SIGN_AREA) {
            signArea(img, platform.posRect);
            signArea(img, platform.doraRect);
        }
        if (SIGN_SMALL) {
            signSmall(img, parall_small, match_result, types, int(platform.posRect.size()));
        }
        imwrite(dest_filename, img);
        if (FLAG_CLOCK) {
            endTime = clock();
            printf_s("Sign: ");
            printf_s("%fs\n", (double)(endTime - startTime) / CLOCKS_PER_SEC);
        }
    }

    if (SHOW_FINALLY) {
        namedWindow("finally", WINDOW_NORMAL);
        imshow("finally", img);
    }

    if (SHOW_CALLBACK || SHOW_RECTS || SHOW_FINALLY) {
        waitKey(0);
    }

    return finallyInfo;
}


vector<vector<string>> Recognition::recognize(Mat image, string platform, string dest_filename,
    string match_filename, string info_filename) {

    FILE *fMatchWrite = NULL;
    FILE *fInfoWrite = NULL;
    if (match_filename != "") {
        fopen_s(&fMatchWrite, match_filename.c_str(), "w");
    }
    if (info_filename != "") {
        fopen_s(&fInfoWrite, info_filename.c_str(), "w");
    }

    int platformID = 0;
    map<string, int> vaildPlatforms = getVaildPlatforms();
    map<string, int>::iterator iter = vaildPlatforms.find(platform);
    if (iter != vaildPlatforms.end()) {
        platformID = iter->second;
    }
    else {
        printf_s("Invaild platform! Set default platform!\n");
    }

    vector<vector<string>> finallyInfo = recognize(Instance(image, to_string(platformID)),
        dest_filename, fMatchWrite, fInfoWrite);

    if (fInfoWrite != NULL) {
        fclose(fInfoWrite);
    }
    if (fMatchWrite != NULL) {
        fclose(fMatchWrite);
    }

    return finallyInfo;
}


vector<vector<string>> Recognition::recognize(Mat image, string platform) {

    return recognize(image, platform, "", "", "");
}


vector<vector<string>> Recognition::recognize(string image_filename, string platform) {

    return recognize(imread(image_filename), platform);
}


int Recognition::test_sign(string source_folder, string dest_folder) {

    if (_access_s(source_folder.c_str(), 0) != 0 || (_access_s(dest_folder.c_str(), 0) != 0
        && _mkdir(dest_folder.c_str()) == -1)) {
        printf_s("Error folder!\n");
        return -1;
    }

    FILE *fMatchWrite = NULL;
    FILE *fInfoWrite = NULL;
    if (FILE_MATCH != NULL) {
        fopen_s(&fMatchWrite, FILE_MATCH, "w");
    }
    if (FILE_INFO != NULL) {
        fopen_s(&fInfoWrite, FILE_INFO, "w");
    }

    struct _finddata_t fileinfo;
    string source_filename, dest_filename;
    string platform;
    intptr_t handle;
    handle = _findfirst((source_folder + "/*.png").c_str(), &fileinfo);
    int count = 0;
    if (!handle) {
        printf_s("Error source folder!\n");
        return -1;
    }
    else {
#ifndef SOURCE_FILENAME
        do {
            source_filename = source_folder + "/" + fileinfo.name;
            dest_filename = dest_folder + "/" + fileinfo.name;
#else
            source_filename = source_folder + "/" + SOURCE_FILENAME;
            dest_filename = dest_folder + "/" + SOURCE_FILENAME;
#endif
            platform = source_filename.substr(source_filename.find(".", 0) - 1, 1);
            recognize(Instance(source_filename, platform), dest_filename, fMatchWrite, fInfoWrite);
            printf_s("%d\n", ++count);
#ifndef SOURCE_FILENAME
        } while (_findnext(handle, &fileinfo) == 0);
#endif
    }

    if (fInfoWrite != NULL) {
        fclose(fInfoWrite);
    }
    if (fMatchWrite != NULL) {
        fclose(fMatchWrite);
    }

    return 0;
}


int main(int argc, char* argv[]) {

    if (argc < 3) {
        printf_s("Need more arguments! Set default!\n");
        const char* argv_const[]{ "Majong.exe", "test/png", "test/sign" };
        for (int i = 0; i < 3; i++) {
            argv[i] = new char[256];
            strcpy_s(argv[i], 256, argv_const[i]);
        }
        argc = 3;
    }

    Recognition recognition;
    recognition.test_sign(argv[1], argv[2]);

    return 0;
}
