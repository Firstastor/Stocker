import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3
import QtQuick.Layouts

Page {
    id: root

    StockPlotPage {
        id: stockPlotPage
        visible: false
    }
    property string stockCode: ""
    property string stockName: ""
    property var stockData: []
    property var historyData: []
    
    function setStockData(code, name) {
        stockCode = code
        stockName = name
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10

        Label {
            text: "股票预测"
            font.bold: true
            font.pixelSize: 16
            Layout.alignment: Qt.AlignLeft
        }        

        Frame {
            Layout.fillWidth: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 10
                
                RowLayout {
                    
                    Label { text: "预测模型:" }
                    ComboBox {
                        id: modelType
                        model: ["LGBM"]
                        Layout.preferredWidth: 150
                    }
                    
                    CheckBox {
                        text: "自动调参"
                        checked: true
                        Layout.alignment: Qt.AlignLeft
                    }
                    
                    CheckBox {
                        text: "注意力机制"
                        checked: true
                        Layout.alignment: Qt.AlignLeft
                    }
                    
                    Item { Layout.fillWidth: true }
                    
                    Button {
                        text: "开始预测"
                        onClicked: runPrediction()
                    }
                }
            }
        }
        
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 15
            
            GroupBox {
                title: "收益预测"
                Layout.fillHeight: true
                Layout.fillWidth: true
            }
            
            GroupBox {
                title: "买卖信号"
                Layout.fillHeight: true
                Layout.fillWidth: true
                
                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10
                    
                    RowLayout {
                        Layout.fillWidth: true
                        
                        Label {
                            text: "当前股票: " + (stockName || "未选择")
                            font.bold: true
                        }
                    }
                    
                    Flickable {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        contentWidth: width
                        contentHeight: signalColumn.height
                        clip: true
                        
                        Column {
                            id: signalColumn
                            width: parent.width
                            spacing: 10
                            
                            // 买入信号标题
                            Label {
                                text: "买入信号"
                                font.bold: true
                                font.pixelSize: 14
                                color: "#aa0000"
                            }
                            
                            // 买入信号列表
                            Repeater {
                                model: buySignals
                                delegate: Row {
                                    spacing: 15
                                    width: parent.width
                                    
                                    Label {
                                        text: "• " + modelData["名称"]
                                        font.pixelSize: 12
                                    }
                                    
                                    Label {
                                        text: "强度: " + modelData["强度"] + "%"
                                        font.pixelSize: 12
                                        color: getStrengthColor(modelData.strength)
                                    }
                                    
                                    Label {
                                        text: modelData["描述"]
                                        font.pixelSize: 12
                                        color: palette.text
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }
                                }
                            }
                            
                            Rectangle {
                                width: parent.width
                                height: 1
                                color: "#e0e0e0"
                                Layout.topMargin: 5
                                Layout.bottomMargin: 5
                            }
                            
                            // 卖出信号标题
                            Label {
                                text: "卖出信号"
                                font.bold: true
                                font.pixelSize: 14
                                color: "#00aa00"
                            }
                            
                            // 卖出信号列表
                            Repeater {
                                model: sellSignals
                                delegate: Row {
                                    spacing: 15
                                    width: parent.width
                                    
                                    Label {
                                        text: "• " + modelData["名称"]
                                        font.pixelSize: 12
                                    }
                                    
                                    Label {
                                        text: "强度: " + modelData["强度"] + "%"
                                        font.pixelSize: 12
                                        color: getStrengthColor(modelData.strength)
                                    }
                                    
                                    Label {
                                        text: modelData["描述"]
                                        font.pixelSize: 12
                                        color: palette.text
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    function runPrediction() {
        console.log("Running prediction for", stockCode, "with model:", modelType.currentText, "days:", predictDays.value)
    }
    // 新增属性
    property var buySignals: []
    property var sellSignals: []

    function getStrengthColor(strength) {
        if (strength >= 80) return "#ff0000"
        if (strength >= 60) return "#ff6600"
        return palette.text
    }

    function refreshSignals() {
        if (!historyData || historyData.length === 0) {
            buySignals = []
            sellSignals = []
            return
        }
        
        var signals = StockCalculate.calculate_trading_signals(historyData)
        if (!signals) {
            buySignals = []
            sellSignals = []
            return
        }
        
        // 使用更安全的属性访问方式
        buySignals = signals["买入信号"] || []
        sellSignals = signals["卖出信号"] || []
        
        // 强制属性变更通知
        buySignals = buySignals.slice() // 创建新数组以触发绑定更新
        sellSignals = sellSignals.slice()
    }

}