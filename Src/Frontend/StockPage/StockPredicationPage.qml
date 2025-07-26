import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts

Page {
    id: root
    property string stockCode: ""
    property string stockName: ""
    property var historyData: []
    
    function setStockData(code, name) {
        stockCode = code
        stockName = name
        loadHistoryData()
    }
    
    function loadHistoryData() {
        historyData = StockGet.get_history_stock_data(stockCode, 240, 365*3) // 3年日线数据
    }
    
    ColumnLayout {
        anchors.fill: parent
        spacing: 10
        
        GroupBox {
            title: "预测参数"
            Layout.fillWidth: true
            Layout.preferredHeight: 80
            
            RowLayout {
                anchors.fill: parent
                
                Label { text: "预测天数:" }
                SpinBox {
                    id: predictDays
                    from: 1
                    to: 30
                    value: 7
                }
                
                Label { text: "预测模型:" }
                ComboBox {
                    id: modelType
                    model: ["LGBM","LSTM"]
                    Layout.fillWidth: true
                }
                
                Button {
                    text: "开始预测"
                    Layout.columnSpan: 3
                    Layout.alignment: Qt.AlignRight
                    onClicked: runPrediction()
                }
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            
            GroupBox {
                title: "预测结果"
                Layout.fillWidth: true
                Layout.fillHeight: true
                
            }
            GroupBox {
                title: "模型评估"
                Layout.fillWidth: true
                Layout.fillHeight: true
            
        }
        }

    }
    
    function runPrediction() {
        console.log("Running prediction for", stockCode, "with model:", modelType.currentText, "days:", predictDays.value)
        // 这里将实现预测逻辑
    }
}