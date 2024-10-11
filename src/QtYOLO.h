#pragma once
#include "ui_QtYOLO.h"
#include <QMainWindow>
#include <QFileDialog>
#include <QImage>
#include <QPixmap>
#include <QMessageBox>
#include <QLabel>
#include <QTextStream>
#include <QDateTime>
#include <QPainter>
#include <opencv2/opencv.hpp>
#include "include/yolov8_onnx.h" 
#include "include/yolov8_obb_onnx.h"
#include "include/yolov8_pose_onnx.h"
#include "include/yolov8_seg_onnx.h"
class QtYOLO : public QMainWindow {
    Q_OBJECT
    
public:
    QtYOLO(QWidget* parent = nullptr);
    ~QtYOLO();
private slots:
    void onSelectFolder();
    void onSelectModel();
    void onDetection();
    void onSegment();
    void onPose();
    void onObb();
private:
    Ui_QtYOLO* ui;
    QLabel* m_imageLabel;
    QString m_folderPath;
    QString m_modelPath;
    
    void updateLog(const QString& message);
    void setupImageDisplay();
    void displayProcessedImage(const cv::Mat& image);
};