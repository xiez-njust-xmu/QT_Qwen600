#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QTextEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QGroupBox>
#include <string>

#include "inference_engine.h"

// Background inference thread
class InferenceWorker : public QThread {
    Q_OBJECT
public:
    explicit InferenceWorker(InferenceEngine* engine, QObject* parent = nullptr)
        : QThread(parent), engine_(engine), stop_(false) {}

    void setRequest(const std::string& input, const std::string& system_prompt) {
        user_input_ = input;
        system_prompt_ = system_prompt;
        stop_ = false;
    }

    void requestStop() { stop_ = true; }

signals:
    void tokenReady(const QString& token);
    void generationFinished();
    void generationError(const QString& error);

protected:
    void run() override {
        if (!engine_ || !engine_->is_initialized()) {
            emit generationError("Engine not initialized");
            return;
        }
        engine_->generate(
            user_input_, system_prompt_,
            [this](const std::string& tok) {
                emit tokenReady(QString::fromUtf8(tok.c_str(), (int)tok.size()));
            },
            &stop_
        );
        emit generationFinished();
    }

private:
    InferenceEngine* engine_;
    std::string user_input_;
    std::string system_prompt_;
    volatile bool stop_;
};

// Main chat window
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(InferenceEngine* engine, QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void onSendClicked();
    void onTokenReceived(const QString& token);
    void onGenerationFinished();
    void onStopClicked();
    void onClearContext();

private:
    void appendUserMessage(const QString& text);
    void startAssistantMessage();

    InferenceEngine* engine_;
    InferenceWorker* worker_;

    QTextEdit* chatDisplay_;
    QLineEdit* inputField_;
    QPushButton* sendBtn_;
    QPushButton* stopBtn_;
    QPushButton* clearBtn_;

    QDoubleSpinBox* tempSpin_;
    QDoubleSpinBox* topPSpin_;
    QSpinBox* topKSpin_;
    QLineEdit* systemPromptEdit_;

    bool generating_;
};

#endif // MAINWINDOW_H
