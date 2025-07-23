import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3
import QtQuick.Layouts

Rectangle {
    id: root
    property int currentScale: 240
    property string stockCode: ""
    property string stockName: ""
    property color textColor: palette.text
    property color upColor: "red"
    property color downColor: "green"
    signal scaleSelected(int newScale)
    implicitHeight: contentLayout.implicitHeight + 20
    color: palette.window

    ColumnLayout {
        id: contentLayout
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            spacing: 20

            Text {
                text: stockCode.slice(2) + " " + stockName
                color: textColor
                font.pixelSize: 18
                font.bold: true
            }

            Text {
                id: currentPrice
                text: "0.00"
                font.pixelSize: 22
                font.bold: true
                color: upColor
            }
            
            Item { Layout.fillWidth: true }

            RowLayout {
                property int currentScale: 240
                property color textColor: palette.text
                property color gridColor: "#e0e0e0"
                height: 40
                spacing: 1

                ButtonGroup { id: scaleGroup }

                Repeater {
                    model: ["日K", "周K", "月K", "5", "15", "30", "60"]
                    Button {
                        text: modelData
                        flat: true
                        checked: [240, 240*5, 7200, 5, 15, 30, 60][index] === root.currentScale
                        ButtonGroup.group: scaleGroup
                        onClicked: {
                            var newScale = [240, 240*5, 7200, 5, 15, 30, 60][index]
                            root.scaleSelected(newScale)
                            root.currentScale = newScale
                        }
                    }
                }
            }
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 6
            columnSpacing: 10
            rowSpacing: 5

            Label { 
                text: "最高" 
                font.bold: true 
                color: textColor 
            }
            Label { 
                id: highPrice
                text: "-" 
                color: textColor 
            }
            Label { 
                text: "最低" 
                font.bold: true 
                color: textColor 
            }
            Label { 
                id: lowPrice
                text: "-" 
                color: textColor 
            }
            Label { 
                text: "今开" 
                font.bold: true 
                color: textColor 
            }
            Label { 
                id: openPrice
                text: "-" 
                color: textColor 
            }
            Label { 
                text: "成交量" 
                font.bold: true 
                color: textColor 
            }
            Label { 
                id: transactionVolume
                text: "-" 
                color: textColor 
            }
            Label { 
                
                text: "成交额" 
                font.bold: true 
                color: textColor 
            }
            Label { 
                id: transactionAmount
                text: "-" 
                color: textColor 
            }
            Label { 
                text: "换手率" 
                font.bold: true 
                color: textColor 
            }
            Label { 
                id: turnoverRate
                text: "-" 
                color: textColor 
            }
        }
    }

    function updateStockData(code, name) {
        stockCode = code
        stockName = name
        
        try {
            for (var i = 0; i < StockInfoData.rowCount(); i++) {
                var item = StockInfoData.get(i)
                if (item.代码 === code) {
                    currentPrice.text = item.最新价
                    currentPrice.color = item.涨跌 >= 0 ? upColor : downColor
                    highPrice.text = item.最高 || "-"
                    lowPrice.text = item.最低 || "-"
                    openPrice.text = item.今开 || "-"
                    transactionVolume.text = item.成交量 || "-"
                    transactionAmount.text = item.成交额 || "-"
                    break
                }
            }
        } catch (error) {
            console.error("更新股票数据错误:", error)
        }
    }

}