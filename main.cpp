#include <QApplication>
#include <QMessageBox>
#include <string>

#include "app_config.h"
#include "inference_engine.h"
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    std::string model_dir = MODEL_DIR;

    InferenceEngine engine;
    InferenceParams params;
    params.temperature = DEFAULT_TEMPERATURE;
    params.top_p = DEFAULT_TOP_P;
    params.top_k = DEFAULT_TOP_K;
    params.enable_thinking = DEFAULT_ENABLE_THINKING;

    if (!engine.init(model_dir, params)) {
        QMessageBox::critical(nullptr, "Error",
            QString("Failed to load model from:\n%1\n\n"
                    "Please check MODEL_DIR in app_config.h")
                .arg(QString::fromStdString(model_dir)));
        return 1;
    }

    MainWindow w(&engine);
    w.show();

    return app.exec();
}
