#pragma once


struct color_callback_data {
    Mat img;
    Mat bgr;
    Mat hsv;
    int hmin;
    int hmax;
    int smin;
    int smax;
    int vmin;
    int vmax;
    int nomask;
    int mask;
    string dstName;
    Mat dst;

    color_callback_data() {}
    color_callback_data(int hmin_, int hmax_, int smin_, int smax_, int vmin_, int vmax_, int nomask_, int mask_, string dstName_) :
        hmin(hmin_), hmax(hmax_), smin(smin_), smax(smax_), vmin(vmin_), vmax(vmax_), nomask(nomask_), mask(mask_), dstName(dstName_) {}
    color_callback_data clone() {
        color_callback_data ccb_copy = color_callback_data();
        ccb_copy.img = img.clone();
        ccb_copy.bgr = bgr.clone();
        ccb_copy.hsv = hsv;
        ccb_copy.hmin = hmin;
        ccb_copy.hmax = hmax;
        ccb_copy.smin = smin;
        ccb_copy.smax = smax;
        ccb_copy.vmin = vmin;
        ccb_copy.vmax = vmax;
        ccb_copy.nomask = nomask;
        ccb_copy.mask = mask;
        ccb_copy.dstName = dstName;
        ccb_copy.dst = dst.clone();
        return ccb_copy;
    }
};


struct MeanStd {
    vector<double> *mean;
    vector<double> *std;
};


class Platform
{
public:
    double resizeRatio;

    color_callback_data color_callback_data1;
    color_callback_data color_callback_data1_othercolor;
    bool hasErode;
    int areaThreshold;
    double otherAreaThresholdRatio;
    double minSideRatio;
    // area

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
    // small

    bool isReversal;
    vector<string> templetNames;

    Platform(double _resizeRatio, color_callback_data _color_callback_data1, color_callback_data _color_callback_data1_othercolor,
        bool _hasErode, int _areaThreshold, double _otherAreaThresholdRatio, double _minSideRatio,
        vector<Vec<double, 5>> _posRect, vector<Vec<double, 5>> _doraRect, double _heightRatio, double _heightRatioDelta, double _lineMaxStd,
        double _lineMinRatio, double _lineMinRatioDelta, double _lineMinRatioDeltaType0, double _lineMaxRatio,
        double _lineMaxRatioDeltaType0, double _otherLineMinRatio, double _otherLineMaxRatio, bool _isReversal, vector<string> _templetNames)
        : resizeRatio(_resizeRatio), color_callback_data1(_color_callback_data1),
        color_callback_data1_othercolor(_color_callback_data1_othercolor), hasErode(_hasErode), areaThreshold(_areaThreshold),
        otherAreaThresholdRatio(_otherAreaThresholdRatio), minSideRatio(_minSideRatio), posRect(_posRect), doraRect(_doraRect),
        heightRatio(_heightRatio), heightRatioDelta(_heightRatioDelta), lineMaxStd(_lineMaxStd), lineMinRatio(_lineMinRatio),
        lineMinRatioDelta(_lineMinRatioDelta), lineMinRatioDeltaType0(_lineMinRatioDeltaType0), lineMaxRatio(_lineMaxRatio),
        lineMaxRatioDeltaType0(_lineMaxRatioDeltaType0), otherLineMinRatio(_otherLineMinRatio), otherLineMaxRatio(_otherLineMaxRatio),
        isReversal(_isReversal), templetNames(_templetNames) {}
};


class Instance
{
public:
    string filename;
    Mat image;
    string platform;

    Instance(string _filename, string _platform) : filename(_filename), platform(_platform) { image = imread(filename); }
    Instance(Mat _image, string _platform) : image(_image), platform(_platform) {}
};


class Recognition
{
public:
    map<string, int> getVaildPlatforms();
    vector<vector<string>> recognize(Mat image, string platform, string dest_filename, string match_filename, string info_filename);
    vector<vector<string>> recognize(Mat image, string platform);
    vector<vector<string>> recognize(string image_filename, string platform);
    int test_sign(string source_folder, string dest_folder);
private:
    void color(Mat &img, Mat &dst, color_callback_data &ccb);
    double getOverlap(const Rect &box1, const Rect &box2);
    vector<Point> getEdgePoint(vector<Point> &contour);
    vector<Vec<double, 5>> find(Mat &img, int areaThreshold, int minSide);
    MeanStd ms(Mat &img, Vec<double, 5> &pa, int type);
    void filter(MeanStd &meanstd, double maxstd);
    vector<int> selectlines(MeanStd &meanstd);
    vector<Vec<double, 5>> morlines(Mat &img, Vec<double, 5> &parall, int fullType, int height, int heightDelta, double maxstd,
        double min_ratio, double min_ratio_delta, double min_ratio_delta_type0, double max_ratio, double max_ratio_delta_type0);
    bool isInParallelogram(Vec2d &point, Vec<double, 5> &parallelogram);
    int range(Rect rect, Mat &img, vector<Vec<double, 5>> &posRect);
    void rotate(Mat &img_rect, int type, bool isReversal);
    vector<Vec<double, 5>> getRects(Mat img, color_callback_data &color_callback_data1, bool hasErode, int areaThreshold, double minSideRatio);
    Rect parall2Rect(Vec<double, 5> &parall, Mat &img);
    double getOverlap(Vec<double, 5> &para1, Vec<double, 5> &para2, Mat &img);
    void getOtherRects(vector<Vec<double, 5>> &paralls, vector<Vec<double, 5>> posRect, Mat img,
        color_callback_data &color_callback_data1_othercolor, bool hasErode, int areaThreshold, double minSideRatio);
    void distHandCards(vector<Vec<double, 5>> &parall_small, vector<int> &types, int handCardsType, int minLimit, int maxLimit);
    void getSmall(vector<Vec<double, 5>> &parall_small, vector<int> &types, vector<Vec<double, 5>> &paralls, Mat &img, vector<Vec<double, 5>> &posRect,
        double heightRatio, double heightRatioDelta, double maxstd, double min_ratio, double min_ratio_delta, double min_ratio_delta_type0,
        double max_ratio, double max_ratio_delta_type0, double otherLineMinRatio, double otherLineMaxRatio);
    void addDora(vector<Vec<double, 5>> &parall_small, vector<int> &types, MatSize size, vector<Vec<double, 5>> &doraRect, int doraType);
    void signSmall(Mat &img, vector<Vec<double, 5>> &parall_small, vector<string> &match_result, vector<int> &types, int handCardsType);
    void signArea(Mat &img, vector<Vec<double, 5>> &posRect);
    Platform getPlatform(string platName);
    vector<string> DNNMatch(Mat img, vector<Vec<double, 5>> &paralls, vector<int> &types, bool isReversal, string model_file, vector<string> &templetNames);
    vector<vector<string>> getFinallyInfo(vector<string> match_result, vector<int> types, int infoCount);
    void createDNNDataset(Mat &img, vector<Vec<double, 5>> &paralls, string dest_filename, vector<int> &types, bool isReversal);
    vector<vector<string>> recognize(Instance instance, string dest_filename, FILE *fMatchWrite, FILE *fInfoWrite, string dataset_filename);
};


#define FLAG_CLOCK true

#define SHOW_CALLBACK false
#define SHOW_RECTS false
#define SHOW_FINALLY false

//#define SOURCE_FILENAME "1_0.png"
//#define SMALL_INDEX 0
//#define SMALL_TYPE 0

#define MODEL_FILE "test/model.pb"
#define MODEL_SCALE 1.0f
#define MODEL_INPWIDTH 64
#define MODEL_INPHEIGHT 64

#define MIN_CARDLIMIT 10

#define SIGN_AREA true
#define SIGN_SMALL true

#define FILE_MATCH "test/match_result.txt"
#define FILE_INFO "test/finallyInfo.txt"


#define LABEL_NAMES { "bai", "bei", "dong", "fa", "nan", "no", "tiao1", "tiao2", \
                        "tiao3", "tiao4", "tiao5", "tiao6", "tiao7", "tiao8", "tiao9", \
                        "tong1", "tong2", "tong3", "tong4", "tong5", "tong6", "tong7", \
                        "tong8", "tong9", "wan1", "wan2", "wan3", "wan4", "wan5", "wan6", \
                        "wan7", "wan8", "wan9", "xi", "zhong" }

#define VAILD_PLATFORMS map<string, int> { \
                            pair<string, int>("中至上饶麻将", 0), \
                            pair<string, int>("腾讯欢乐麻将", 1), \
                            pair<string, int>("大唐麻将", 2), \
                            pair<string, int>("微乐吃鸡麻将", 3), \
                            pair<string, int>("麻将来了", 4), \
                            pair<string, int>("JJ麻将", 5), \
                            pair<string, int>("欢乐真人麻将", 6), \
                            /*pair<string, int>("熊猫四川麻将", 7),*/ \
                            pair<string, int>("欢乐麻将全集（博雅）", 8), \
                            pair<string, int>("四川麻将（血战到底）", 9) }

#define PARMS_PLAT_0 resizeRatio = 0.6; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 183, 250, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(54, 96, 61, 155, 0, 255, 0, -1, "dora"); \
                    hasErode = false; \
                    areaThreshold = 300; \
                    otherAreaThresholdRatio = 1; \
                    minSideRatio = 0.015; \
                    posRect = { Vec<double, 5>(0.03, 0.83, 0.95, 0.16, 0.00), Vec<double, 5>(0.32, 0.60, 0.37, 0.21, 0.00), \
                                Vec<double, 5>(0.12, 0.14, 0.04, 0.53, 0.00), Vec<double, 5>(0.17, 0.25, 0.11, 0.50, 0.00), \
                                Vec<double, 5>(0.28, 0.03, 0.48, 0.11, 0.00), Vec<double, 5>(0.32, 0.15, 0.37, 0.21, 0.00), \
                                Vec<double, 5>(0.86, 0.24, 0.04, 0.53, 0.00), Vec<double, 5>(0.72, 0.25, 0.11, 0.50, 0.00) }; \
                    doraRect = { Vec<double, 5>(0.935, 0.04, 0.04, 0.095, 0.00) }; \
                    heightRatio = 0.07; \
                    heightRatioDelta = 0.0; \
                    lineMaxStd = 30.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.0; \
                    lineMinRatioDeltaType0 = 0.0; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = 0.0; \
                    otherLineMinRatio = 0.05; \
                    otherLineMaxRatio = 0.08; \
                    isReversal = true;

#define PARMS_PLAT_1 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 183, 255, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(0, 100, 0, 25, 115, 169, 0, -1, "othercolor"); \
                    hasErode = false; \
                    areaThreshold = 200; \
                    minSideRatio = 0.010; \
                    posRect = { Vec<double, 5>(0.03, 0.83, 0.95, 0.16, 0.00), Vec<double, 5>(0.41, 0.53, 0.30, 0.21, 0.00), \
                                Vec<double, 5>(0.17, 0.11, 0.05, 0.60, -0.10), Vec<double, 5>(0.25, 0.28, 0.16, 0.40, 0.00), \
                                Vec<double, 5>(0.19, 0.01, 0.46, 0.06, 0.00), Vec<double, 5>(0.42, 0.10, 0.16, 0.19, 0.00), \
                                Vec<double, 5>(0.79, 0.11, 0.05, 0.64, 0.10), Vec<double, 5>(0.59, 0.24, 0.16, 0.28, 0.00) }; \
                    heightRatio = 0.06; \
                    heightRatioDelta = 0.03; \
                    lineMaxStd = 30.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.1; \
                    lineMinRatioDeltaType0 = 0.1; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = 0.4; \
                    otherLineMinRatio = 0.04; \
                    otherLineMaxRatio = 0.08; \
                    isReversal = true;

#define PARMS_PLAT_2 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 183, 255, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(0, 100, 0, 25, 115, 169, 0, -1, "no"); \
                    hasErode = false; \
                    areaThreshold = 1000; \
                    minSideRatio = 0.015; \
                    posRect = { Vec<double, 5>(0.03, 0.83, 0.95, 0.16, 0.00), Vec<double, 5>(0.32, 0.60, 0.37, 0.23, 0.00), \
                                Vec<double, 5>(0.10, 0.14, 0.04, 0.63, 0.00), Vec<double, 5>(0.14, 0.25, 0.11, 0.50, 0.00), \
                                Vec<double, 5>(0.34, 0.12, 0.38, 0.09, 0.00), Vec<double, 5>(0.34, 0.21, 0.35, 0.21, 0.00), \
                                Vec<double, 5>(0.86, 0.20, 0.04, 0.60, 0.00), Vec<double, 5>(0.74, 0.22, 0.12, 0.55, 0.00) }; \
                    heightRatio = 0.06; \
                    heightRatioDelta = 0.0; \
                    lineMaxStd = 30.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.0; \
                    lineMinRatioDeltaType0 = 0.0; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = 0.0; \
                    otherLineMinRatio = 0.05; \
                    otherLineMaxRatio = 0.08; \
                    isReversal = false;

#define PARMS_PLAT_3 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 183, 255, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(0, 100, 0, 25, 115, 169, 0, -1, "no"); \
                    hasErode = false; \
                    areaThreshold = 200; \
                    minSideRatio = 0.010; \
                    posRect = { Vec<double, 5>(0.00, 0.80, 1.00, 0.20, 0.00), Vec<double, 5>(0.36, 0.55, 0.28, 0.25, 0.00), \
                                Vec<double, 5>(0.21, 0.05, 0.04, 0.70, -0.07), Vec<double, 5>(0.23, 0.18, 0.15, 0.44, -0.03), \
                                Vec<double, 5>(0.25, 0.03, 0.45, 0.08, 0.00), Vec<double, 5>(0.38, 0.11, 0.25, 0.21, 0.00), \
                                Vec<double, 5>(0.76, 0.14, 0.05, 0.66, 0.06), Vec<double, 5>(0.63, 0.20, 0.14, 0.50, 0.02) }; \
                    heightRatio = 0.07; \
                    heightRatioDelta = 0.03; \
                    lineMaxStd = 30.0; \
                    lineMinRatio = 0.45; \
                    lineMinRatioDelta = 0.0; \
                    lineMinRatioDeltaType0 = -0.1; \
                    lineMaxRatio = 0.9; \
                    lineMaxRatioDeltaType0 = -0.1; \
                    otherLineMinRatio = 0.05; \
                    otherLineMaxRatio = 0.09; \
                    isReversal = false;

#define PARMS_PLAT_4 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 183, 255, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(0, 100, 0, 25, 115, 169, 0, -1, "no"); \
                    hasErode = false; \
                    areaThreshold = 200; \
                    minSideRatio = 0.010; \
                    posRect = { Vec<double, 5>(0.16, 0.85, 0.75, 0.13, 0.00), Vec<double, 5>(0.37, 0.67, 0.28, 0.10, 0.00), \
                                Vec<double, 5>(0.21, 0.32, 0.04, 0.50, -0.10), Vec<double, 5>(0.30, 0.42, 0.09, 0.30, -0.03), \
                                Vec<double, 5>(0.30, 0.26, 0.45, 0.06, 0.00), Vec<double, 5>(0.38, 0.34, 0.23, 0.09, 0.00), \
                                Vec<double, 5>(0.76, 0.34, 0.04, 0.45, 0.09), Vec<double, 5>(0.61, 0.40, 0.08, 0.26, 0.03) }; \
                    heightRatio = 0.07; \
                    heightRatioDelta = 0.05; \
                    lineMaxStd = 40.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.205; \
                    lineMinRatioDeltaType0 = 0.1; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = 0.4; \
                    otherLineMinRatio = 0.05; \
                    otherLineMaxRatio = 0.075; \
                    isReversal = true; \

#define PARMS_PLAT_5 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 39, 220, 255, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(0, 100, 0, 25, 115, 169, 0, -1, "no"); \
                    hasErode = true; \
                    areaThreshold = 200; \
                    minSideRatio = 0.014; \
                    posRect = { Vec<double, 5>(0.00, 0.78, 0.94, 0.16, 0.00), Vec<double, 5>(0.33, 0.53, 0.34, 0.25, 0.00), \
                                Vec<double, 5>(0.21, 0.18, 0.04, 0.60, -0.17), Vec<double, 5>(0.22, 0.26, 0.16, 0.38, -0.08), \
                                Vec<double, 5>(0.30, 0.17, 0.40, 0.06, 0.00), Vec<double, 5>(0.36, 0.23, 0.27, 0.17, 0.00), \
                                Vec<double, 5>(0.75, 0.18, 0.04, 0.58, 0.17), Vec<double, 5>(0.63, 0.27, 0.15, 0.40, 0.09) }; \
                    heightRatio = 0.07; \
                    heightRatioDelta = 0.03; \
                    lineMaxStd = 40.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.2; \
                    lineMinRatioDeltaType0 = 0.0; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = -0.2; \
                    otherLineMinRatio = 0.05; \
                    otherLineMaxRatio = 0.08; \
                    isReversal = false;

#define PARMS_PLAT_6 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 183, 255, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(0, 360, 0, 25, 131, 150, 0, -1, "othercolor"); \
                    hasErode = true; \
                    areaThreshold = 200; \
                    minSideRatio = 0.010; \
                    posRect = { Vec<double, 5>(0.10, 0.83, 0.88, 0.16, 0.00), Vec<double, 5>(0.39, 0.50, 0.22, 0.29, 0.00), \
                                Vec<double, 5>(0.12, 0.04, 0.05, 0.65, -0.12), Vec<double, 5>(0.20, 0.22, 0.21, 0.32, -0.03), \
                                Vec<double, 5>(0.30, 0.02, 0.48, 0.06, 0.00), Vec<double, 5>(0.41, 0.08, 0.18, 0.18, 0.00), \
                                Vec<double, 5>(0.84, 0.10, 0.05, 0.58, 0.10), Vec<double, 5>(0.59, 0.22, 0.20, 0.30, 0.03) }; \
                    heightRatio = 0.07; \
                    heightRatioDelta = 0.035; \
                    lineMaxStd = 30.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.1; \
                    lineMinRatioDeltaType0 = -0.1; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = -0.2; \
                    otherLineMinRatio = 0.05; \
                    otherLineMaxRatio = 0.07; \
                    isReversal = false;

#define PARMS_PLAT_7 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 183, 250, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(0, 100, 0, 25, 115, 169, 0, -1, "no"); \
                    hasErode = false; \
                    areaThreshold = 1000; \
                    minSideRatio = 0.015; \
                    posRect = { Vec<double, 5>(0.03, 0.83, 0.95, 0.16, 0.00), Vec<double, 5>(0.32, 0.60, 0.37, 0.21, 0.00), \
                                Vec<double, 5>(0.19, 0.14, 0.04, 0.64, 0.00), Vec<double, 5>(0.24, 0.19, 0.11, 0.50, 0.00), \
                                Vec<double, 5>(0.28, 0.10, 0.44, 0.08, 0.00), Vec<double, 5>(0.36, 0.19, 0.32, 0.21, 0.00), \
                                Vec<double, 5>(0.85, 0.12, 0.04, 0.65, 0.00), Vec<double, 5>(0.71, 0.25, 0.13, 0.50, 0.00) }; \
                    heightRatio = 0.07; \
                    heightRatioDelta = 0.0; \
                    lineMaxStd = 30.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.0; \
                    lineMinRatioDeltaType0 = 0.0; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = 0.0; \
                    otherLineMinRatio = 0.05; \
                    otherLineMaxRatio = 0.07; \
                    isReversal = true;

#define PARMS_PLAT_8 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 209, 255, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(0, 360, 0, 25, 110, 130, 0, -1, "othercolor"); \
                    hasErode = false; \
                    areaThreshold = 200; \
                    minSideRatio = 0.010; \
                    posRect = { Vec<double, 5>(0.07, 0.83, 0.80, 0.16, 0.00), Vec<double, 5>(0.41, 0.51, 0.20, 0.21, 0.00), \
                                Vec<double, 5>(0.18, 0.10, 0.05, 0.53, -0.10), Vec<double, 5>(0.30, 0.28, 0.13, 0.30, -0.03), \
                                Vec<double, 5>(0.30, 0.03, 0.50, 0.06, 0.00), Vec<double, 5>(0.42, 0.10, 0.15, 0.21, 0.00), \
                                Vec<double, 5>(0.78, 0.10, 0.05, 0.64, 0.11), Vec<double, 5>(0.57, 0.24, 0.19, 0.27, 0.03) }; \
                    heightRatio = 0.048; \
                    heightRatioDelta = 0.018; \
                    lineMaxStd = 40.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.1; \
                    lineMinRatioDeltaType0 = -0.3; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = -0.2; \
                    otherLineMinRatio = 0.04; \
                    otherLineMaxRatio = 0.07; \
                    isReversal = true;

#define PARMS_PLAT_9 resizeRatio = 1.0; \
                    color_callback_data1 = color_callback_data(0, 360, 0, 72, 183, 255, 0, -1, "color"); \
                    color_callback_data1_othercolor = color_callback_data(234, 360, 0, 96, 91, 122, 0, -1, "othercolor"); \
                    hasErode = false; \
                    areaThreshold = 200; \
                    minSideRatio = 0.010; \
                    posRect = { Vec<double, 5>(0.00, 0.84, 1.00, 0.16, 0.00), Vec<double, 5>(0.40, 0.52, 0.26, 0.23, 0.00), \
                                Vec<double, 5>(0.18, 0.10, 0.04, 0.60, -0.09), Vec<double, 5>(0.25, 0.30, 0.16, 0.30, -0.02), \
                                Vec<double, 5>(0.25, 0.03, 0.42, 0.08, 0.00), Vec<double, 5>(0.42, 0.11, 0.16, 0.21, 0.00), \
                                Vec<double, 5>(0.78, 0.14, 0.04, 0.62, 0.10), Vec<double, 5>(0.58, 0.22, 0.16, 0.30, 0.02) }; \
                    heightRatio = 0.049; \
                    heightRatioDelta = 0.025; \
                    lineMaxStd = 30.0; \
                    lineMinRatio = 0.5; \
                    lineMinRatioDelta = 0.2; \
                    lineMinRatioDeltaType0 = -0.1; \
                    lineMaxRatio = 0.8; \
                    lineMaxRatioDeltaType0 = 0.0; \
                    otherLineMinRatio = 0.04; \
                    otherLineMaxRatio = 0.08; \
                    isReversal = false;

#define PARMS_PLAT(platName) if (platName == "0") { PARMS_PLAT_0 } \
                                else if (platName == "1") { PARMS_PLAT_1 } \
                                else if (platName == "2") { PARMS_PLAT_2 } \
                                else if (platName == "3") { PARMS_PLAT_3 } \
                                else if (platName == "4") { PARMS_PLAT_4 } \
                                else if (platName == "5") { PARMS_PLAT_5 } \
                                else if (platName == "6") { PARMS_PLAT_6 } \
                                else if (platName == "7") { PARMS_PLAT_7 } \
                                else if (platName == "8") { PARMS_PLAT_8 } \
                                else if (platName == "9") { PARMS_PLAT_9 } \
                                else { PARMS_PLAT_0 }
