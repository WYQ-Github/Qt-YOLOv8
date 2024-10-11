#include "QtYOLO.h"


QtYOLO::QtYOLO(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui_QtYOLO)
{
    ui->setupUi(this);
    connect(ui->slectFolder, &QPushButton::clicked, this, &QtYOLO::onSelectFolder);
    connect(ui->modelBtn, &QPushButton::clicked, this, &QtYOLO::onSelectModel);
    connect(ui->Objdet, &QPushButton::clicked, this, &QtYOLO::onDetection);
    connect(ui->SegBtn, &QPushButton::clicked, this, &QtYOLO::onSegment);
    connect(ui->ObbBtn, &QPushButton::clicked, this, &QtYOLO::onSegment);
    connect(ui->PoseBtn, &QPushButton::clicked, this, &QtYOLO::onPose);
    setupImageDisplay();
}

QtYOLO::~QtYOLO()
{
    delete ui; 
}

void QtYOLO::updateLog(const QString& message)
{
    ui->textEdit->append(message);
}


void QtYOLO::setupImageDisplay()
{
    QScrollArea* scrollArea = ui->scrollArea;

    m_imageLabel = new QLabel(scrollArea);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    m_imageLabel->setScaledContents(true);

    scrollArea->setWidget(m_imageLabel);
    scrollArea->setWidgetResizable(true);
}

void QtYOLO::displayProcessedImage(const cv::Mat& image)
{
    if (image.empty()) {
        updateLog("无法加载图片");
        return;
    }
    
    cv::Mat rgb_image;
    // 如果是灰度图，转换为RGB
    if (image.channels() == 1) {
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
    } 
    // 如果是彩色图，从BGR转换为RGB
    else if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else {
        updateLog("不支持的图像格式");
        return;
    }

    QImage qimg(rgb_image.data, 
                rgb_image.cols, 
                rgb_image.rows, 
                static_cast<int>(rgb_image.step),
                QImage::Format_RGB888);
    
    QPixmap pixmap = QPixmap::fromImage(qimg);
    m_imageLabel->setPixmap(pixmap.scaled(m_imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    m_imageLabel->setAlignment(Qt::AlignCenter);
    
    QApplication::processEvents();
}

void QtYOLO::onSelectFolder()
{
    m_folderPath = QFileDialog::getExistingDirectory(this, "选择图片文件夹");
    if (!m_folderPath.isEmpty()) {
        updateLog("已选择文件夹: " + m_folderPath);
    }
}
void QtYOLO::onSelectModel()
{
    m_modelPath = QFileDialog::getOpenFileName(this,"选择模型文件", "", "ONNX文件 (*.onnx)");
    if(!m_modelPath.isEmpty()) {
        updateLog("已选择模型文件: " + m_modelPath);
    }
}

// 目标检测
void QtYOLO::onDetection()
{   
    Yolov8Onnx m_v8Det; 
    if(m_folderPath.isEmpty() || m_modelPath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择图片文件夹和模型文件");
        return;
    }

    m_v8Det.setLogCallback(([this](const std::string& message) {
        updateLog(QString::fromStdString(message));
    }));

    m_v8Det.ReadModel(m_modelPath.toStdString(), true);

    std::vector<std::string> imagePathList;
    
    cv::glob(m_folderPath.toStdString() + "/*.jpg", imagePathList);

	std::vector<cv::Scalar> color;
	std::srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}

    for (const auto& imagePath : imagePathList) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error reading image: " << imagePath << std::endl;
            continue;
        }
        displayProcessedImage(image);
        std::vector<OutputParams> result;
        auto start = std::chrono::system_clock::now();
        m_v8Det.OnnxDetect(image, result);
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        updateLog(QString("Cost: %1 ms").arg(tc, 0, 'f', 4)); 

        DrawPred(image,result,m_v8Det._className,color);
        displayProcessedImage(image);        
    }

    imagePathList.clear();
    m_folderPath.clear();
    m_modelPath.clear();
}

// 实例分割
void QtYOLO::onSegment()
{
    if(m_folderPath.isEmpty() || m_modelPath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择图片文件夹和模型文件");
        return;
    }

    Yolov8SegOnnx m_v8Seg;
    m_v8Seg.setLogCallback(([this](const std::string& message) {
        updateLog(QString::fromStdString(message));
    }));

    m_v8Seg.ReadModel(m_modelPath.toStdString(), true);

    std::vector<std::string> imagePathList;
    
    cv::glob(m_folderPath.toStdString() + "/*.jpg", imagePathList);

	std::vector<cv::Scalar> color;
	std::srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}

    for (const auto& imagePath : imagePathList) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error reading image: " << imagePath << std::endl;
            continue;
        }
        displayProcessedImage(image);
        std::vector<OutputParams> result;
        auto start = std::chrono::system_clock::now();
        m_v8Seg.OnnxDetect(image, result);
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        updateLog(QString("Cost: %1 ms").arg(tc, 0, 'f', 4)); 

        DrawPred(image,result,m_v8Seg._className,color);
        displayProcessedImage(image);        
    }

    imagePathList.clear();
    m_folderPath.clear();
    m_modelPath.clear();
}

// 姿态检测
void QtYOLO::onPose()
{
    if(m_folderPath.isEmpty() || m_modelPath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择图片文件夹和模型文件");
        return;
    }  

    Yolov8PoseOnnx m_v8Pose;

    m_v8Pose.setLogCallback(([this](const std::string& message) {
        updateLog(QString::fromStdString(message));
    }));

    m_v8Pose.ReadModel(m_modelPath.toStdString(), true);

    std::vector<std::string> imagePathList;
    
    cv::glob(m_folderPath.toStdString() + "/*.jpg", imagePathList);

	std::vector<cv::Scalar> color;
	std::srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}

    for (const auto& imagePath : imagePathList) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error reading image: " << imagePath << std::endl;
            continue;
        }
        displayProcessedImage(image);
        std::vector<OutputParams> result;
        PoseParams m_PoseParams;
        auto start = std::chrono::system_clock::now();
        m_v8Pose.OnnxDetect(image, result);
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        updateLog(QString("Cost: %1 ms").arg(tc, 0, 'f', 4)); 

        DrawPredPose(image,result,m_PoseParams);
        displayProcessedImage(image);        
    }

    imagePathList.clear();
    m_folderPath.clear();
    m_modelPath.clear();
}

// 旋转框检测
void QtYOLO::onObb()
{
    if(m_folderPath.isEmpty() || m_modelPath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择图片文件夹和模型文件");
        return;
    }


    Yolov8ObbOnnx m_v8Obb;

    m_v8Obb.setLogCallback(([this](const std::string& message) {
        updateLog(QString::fromStdString(message));
    }));

    m_v8Obb.ReadModel(m_modelPath.toStdString(), true);

    std::vector<std::string> imagePathList;
    
    cv::glob(m_folderPath.toStdString() + "/*.jpg", imagePathList);

	std::vector<cv::Scalar> color;
	std::srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}

    for (const auto& imagePath : imagePathList) {
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Error reading image: " << imagePath << std::endl;
            continue;
        }
        displayProcessedImage(image);
        std::vector<OutputParams> result;
        auto start = std::chrono::system_clock::now();
        m_v8Obb.OnnxDetect(image, result);
        auto end = std::chrono::system_clock::now();
        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        updateLog(QString("Cost: %1 ms").arg(tc, 0, 'f', 4)); 

        DrawPred(image,result,m_v8Obb._className,color);
        displayProcessedImage(image);        
    }

    imagePathList.clear();
    m_folderPath.clear();
    m_modelPath.clear();
}
