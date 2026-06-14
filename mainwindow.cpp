#include "mainwindow.h"
#include "app_config.h"

MainWindow::MainWindow(InferenceEngine* engine, QWidget* parent)
    : QMainWindow(parent), engine_(engine), generating_(false)
{
    setWindowTitle("Qwen600 Chat");
    resize(750, 600);

    QWidget* central = new QWidget(this);
    setCentralWidget(central);

    QVBoxLayout* mainLayout = new QVBoxLayout(central);

    // Chat display area
    chatDisplay_ = new QTextEdit(this);
    chatDisplay_->setReadOnly(true);
    chatDisplay_->setStyleSheet(
        "QTextEdit { background: #1e1e2e; color: #cdd6f4; "
        "font-family: 'Consolas', 'Microsoft YaHei UI'; font-size: 13px; "
        "padding: 10px; border-radius: 4px; }");
    mainLayout->addWidget(chatDisplay_, 1);

    // Input area
    QHBoxLayout* inputLayout = new QHBoxLayout();
    inputField_ = new QLineEdit(this);
    inputField_->setPlaceholderText(QString::fromUtf8("Type your message..."));
    inputField_->setStyleSheet(
        "QLineEdit { padding: 8px; font-size: 13px; border-radius: 4px; }");

    sendBtn_ = new QPushButton(QString::fromUtf8("Send"), this);
    stopBtn_ = new QPushButton(QString::fromUtf8("Stop"), this);
    stopBtn_->setEnabled(false);
    clearBtn_ = new QPushButton(QString::fromUtf8("Clear"), this);

    inputLayout->addWidget(inputField_, 1);
    inputLayout->addWidget(sendBtn_);
    inputLayout->addWidget(stopBtn_);
    inputLayout->addWidget(clearBtn_);
    mainLayout->addLayout(inputLayout);

    // Settings panel
    QGroupBox* settingsGroup = new QGroupBox("Parameters", this);
    QHBoxLayout* settingsLayout = new QHBoxLayout(settingsGroup);

    settingsLayout->addWidget(new QLabel("Temp:"));
    tempSpin_ = new QDoubleSpinBox(this);
    tempSpin_->setRange(0.0, 2.0);
    tempSpin_->setSingleStep(0.1);
    tempSpin_->setValue(DEFAULT_TEMPERATURE);
    settingsLayout->addWidget(tempSpin_);

    settingsLayout->addWidget(new QLabel("Top-P:"));
    topPSpin_ = new QDoubleSpinBox(this);
    topPSpin_->setRange(0.0, 1.0);
    topPSpin_->setSingleStep(0.05);
    topPSpin_->setValue(DEFAULT_TOP_P);
    settingsLayout->addWidget(topPSpin_);

    settingsLayout->addWidget(new QLabel("Top-K:"));
    topKSpin_ = new QSpinBox(this);
    topKSpin_->setRange(1, 100);
    topKSpin_->setValue(DEFAULT_TOP_K);
    settingsLayout->addWidget(topKSpin_);

    settingsLayout->addWidget(new QLabel("System:"));
    systemPromptEdit_ = new QLineEdit(QString::fromUtf8(DEFAULT_SYSTEM_PROMPT), this);
    settingsLayout->addWidget(systemPromptEdit_, 1);

    mainLayout->addWidget(settingsGroup);

    // Worker thread
    worker_ = new InferenceWorker(engine_, this);
    connect(worker_, &InferenceWorker::tokenReady,
            this, &MainWindow::onTokenReceived, Qt::QueuedConnection);
    connect(worker_, &InferenceWorker::generationFinished,
            this, &MainWindow::onGenerationFinished, Qt::QueuedConnection);

    // Button connections
    connect(sendBtn_, &QPushButton::clicked, this, &MainWindow::onSendClicked);
    connect(stopBtn_, &QPushButton::clicked, this, &MainWindow::onStopClicked);
    connect(clearBtn_, &QPushButton::clicked, this, &MainWindow::onClearContext);
    connect(inputField_, &QLineEdit::returnPressed, this, &MainWindow::onSendClicked);
}

MainWindow::~MainWindow() {
    if (worker_->isRunning()) {
        worker_->requestStop();
        worker_->wait(3000);
    }
}

void MainWindow::onSendClicked() {
    if (generating_) return;
    QString text = inputField_->text().trimmed();
    if (text.isEmpty()) return;

    appendUserMessage(text);
    inputField_->clear();

    engine_->update_sampler(
        (float)tempSpin_->value(),
        (float)topPSpin_->value(),
        topKSpin_->value());

    worker_->setRequest(
        text.toUtf8().constData(),
        systemPromptEdit_->text().toUtf8().constData());

    generating_ = true;
    sendBtn_->setEnabled(false);
    stopBtn_->setEnabled(true);
    inputField_->setEnabled(false);
    startAssistantMessage();
    worker_->start();
}

void MainWindow::onTokenReceived(const QString& token) {
    QTextCursor cursor = chatDisplay_->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertText(token);
    chatDisplay_->setTextCursor(cursor);
    chatDisplay_->ensureCursorVisible();
}

void MainWindow::onGenerationFinished() {
    generating_ = false;
    sendBtn_->setEnabled(true);
    stopBtn_->setEnabled(false);
    inputField_->setEnabled(true);
    inputField_->setFocus();

    QTextCursor cursor = chatDisplay_->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.insertText("\n\n");
    chatDisplay_->setTextCursor(cursor);
}

void MainWindow::onStopClicked() {
    worker_->requestStop();
}

void MainWindow::onClearContext() {
    engine_->reset_context();
    chatDisplay_->clear();
    chatDisplay_->append(QString::fromUtf8("<i>[Context cleared]</i>\n"));
}

void MainWindow::appendUserMessage(const QString& text) {
    chatDisplay_->append(
        QString("<b style='color:#89b4fa;'>You:</b> %1\n").arg(text.toHtmlEscaped()));
}

void MainWindow::startAssistantMessage() {
    chatDisplay_->append(QString::fromUtf8("<b style='color:#a6e3a1;'>Assistant:</b> "));
}