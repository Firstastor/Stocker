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
        
        // 重置所有字段为默认值
        currentPrice.text = "0.00"
        currentPrice.color = textColor
        highPrice.text = "-"
        lowPrice.text = "-"
        openPrice.text = "-"
        transactionVolume.text = "-"
        transactionAmount.text = "-"
        turnoverRate.text = "-"
        
        // 使用 StockGet 获取股票数据
        var stockData = StockGet.get_stock_data()
        
        if (!stockData || stockData.length === 0) {
            console.log("未获取到股票数据")
            return
        }
        
        // 查找匹配的股票数据
        var found = false
        for (var i = 0; i < stockData.length; i++) {
            var item = stockData[i]
            if (item.代码 === code) {
                found = true
                
                // 更新最新价和颜色
                if (item.最新价 !== undefined) {
                    currentPrice.text = parseFloat(item.最新价).toFixed(2)
                    currentPrice.color = item.涨跌 >= 0 ? upColor : downColor
                }
                
                // 更新其他字段
                if (item.最高 !== undefined) {
                    highPrice.text = parseFloat(item.最高).toFixed(2)
                }
                
                if (item.最低 !== undefined) {
                    lowPrice.text = parseFloat(item.最低).toFixed(2)
                }
                
                if (item.今开 !== undefined) {
                    openPrice.text = parseFloat(item.今开).toFixed(2)
                }
                
                if (item.成交量 !== undefined) {
                    transactionVolume.text = formatNumber(item.成交量)
                }
                
                if (item.成交额 !== undefined) {
                    transactionAmount.text = formatNumber(item.成交额) + "万"
                }
                
                break
            }
        }
        
        if (!found) {
            console.log("未找到股票代码对应的数据:", code)
        }
    }

    function formatNumber(num) {
        if (num >= 100000000) {
            return (num / 100000000).toFixed(2) + "亿"
        } else if (num >= 10000) {
            return (num / 10000).toFixed(2) + "万"
        }
        return num.toString()
    }

}