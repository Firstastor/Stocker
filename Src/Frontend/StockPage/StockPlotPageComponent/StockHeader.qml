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
    property color upColor: "#e74c3c"  // 红色
    property color downColor: "#2ecc71" // 绿色
    property color flatColor: textColor // 平盘颜色
    property var stockData: []
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

            ColumnLayout {
                spacing: 2
                Text {
                    id: currentPrice
                    text: "0.00"
                    font.pixelSize: 22
                    font.bold: true
                    color: flatColor
                }
                Text {
                    id: priceChange
                    text: "+0.00 (+0.00%)"
                    font.pixelSize: 14
                    color: flatColor
                }
            }
            
            Item { Layout.fillWidth: true }

            RowLayout {
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
            columns: 8
            columnSpacing: 10
            rowSpacing: 5

            // 第一行
            Label { text: "最高"; font.bold: true; color: textColor }
            Label { id: highPrice; text: "-"; color: textColor }
            Label { text: "最低"; font.bold: true; color: textColor }
            Label { id: lowPrice; text: "-"; color: textColor }
            Label { text: "今开"; font.bold: true; color: textColor }
            Label { id: openPrice; text: "-"; color: textColor }
            Label { text: "昨收"; font.bold: true; color: textColor }
            Label { id: prevClose; text: "-"; color: textColor }

            // 第二行
            Label { text: "市盈率"; font.bold: true; color: textColor }
            Label { id: peRatio; text: "-"; color: textColor }
            Label { text: "市净率"; font.bold: true; color: textColor }
            Label { id: pbRatio; text: "-"; color: textColor }
            Label { text: "总市值"; font.bold: true; color: textColor }
            Label { id: totalMarketCap; text: "-"; color: textColor }
            Label { text: "流通市值"; font.bold: true; color: textColor }
            Label { id: circulatingMarketCap; text: "-"; color: textColor }

            // 第三行
            Label { text: "成交量"; font.bold: true; color: textColor }
            Label { id: transactionVolume; text: "-"; color: textColor }
            Label { text: "成交额"; font.bold: true; color: textColor }
            Label { id: transactionAmount; text: "-"; color: textColor }
            Label { text: "换手率"; font.bold: true; color: textColor }
            Label { id: turnoverRate; text: "-"; color: textColor }
            Label { text: "更新时间"; font.bold: true; color: textColor }
            Label { id: updateTime; text: "-"; color: textColor }
        }
    }

    function updateStockData(code, name) {
        stockCode = code
        stockName = name
        
        // 重置所有字段为默认值
        resetFields()
        
        // 查找匹配的股票数据
        var found = false
        for (var i = 0; i < stockData.length; i++) {
            var item = stockData[i]
            if (item.代码 === code) {
                found = true
                updateAllFields(item)
                break
            }
        }
        
        if (!found) {
            console.log("未找到股票代码对应的数据:", code)
        }
    }

    function resetFields() {
        currentPrice.text = "0.00"
        currentPrice.color = flatColor
        priceChange.text = "+0.00 (+0.00%)"
        priceChange.color = flatColor
        highPrice.text = "-"
        lowPrice.text = "-"
        openPrice.text = "-"
        prevClose.text = "-"
        transactionVolume.text = "-"
        transactionAmount.text = "-"
        turnoverRate.text = "-"
        peRatio.text = "-"
        pbRatio.text = "-"
        totalMarketCap.text = "-"
        circulatingMarketCap.text = "-"
    }

    function updateAllFields(item) {
        // 价格和涨跌幅
        if (item.最新价 !== undefined) {
            currentPrice.text = parseFloat(item.最新价).toFixed(2)
            const change = item.涨跌 || 0
            const changePercent = item.涨幅 || 0
            const isUp = change > 0
            const isDown = change < 0
            
            currentPrice.color = isUp ? upColor : isDown ? downColor : flatColor
            priceChange.text = `${change >= 0 ? "+" : ""}${change.toFixed(2)} (${changePercent.toFixed(2)}%)`
            priceChange.color = currentPrice.color
        }
        
        // 价格相关
        if (item.最高 !== undefined) highPrice.text = parseFloat(item.最高).toFixed(2)
        if (item.最低 !== undefined) lowPrice.text = parseFloat(item.最低).toFixed(2)
        if (item.今开 !== undefined) openPrice.text = parseFloat(item.今开).toFixed(2)
        if (item.昨收 !== undefined) prevClose.text = parseFloat(item.昨收).toFixed(2)
        
        // 成交量/额
        if (item.成交量 !== undefined) transactionVolume.text = formatNumber(item.成交量, true)
        if (item.成交额 !== undefined) transactionAmount.text = formatNumber(item.成交额, true)
        
        // 比率数据
        if (item.换手率 !== undefined) turnoverRate.text = parseFloat(item.换手率).toFixed(2) + "%"
        if (item.市盈率 !== undefined) peRatio.text = parseFloat(item.市盈率).toFixed(2)
        if (item.市净率 !== undefined) pbRatio.text = parseFloat(item.市净率).toFixed(2)
        
        // 市值数据
        if (item.总市值 !== undefined) totalMarketCap.text = formatNumber(item.总市值)
        if (item.流通市值 !== undefined) circulatingMarketCap.text = formatNumber(item.流通市值)

        // 更新时间
        if (item.更新时间 !== undefined) updateTime.text = item.更新时间
        else updateTime.text = new Date().toLocaleTimeString() // 默认使用当前时间
    }

    function formatNumber(num, isVolume = false) {
        if (num === undefined || num === null) return "-"
        
        // 处理成交量/成交额(单位:万/亿)
        if (isVolume) {
            if (num >= 100000000) {
                return (num / 100000000).toFixed(2) + "亿"
            } else if (num >= 10000) {
                return (num / 10000).toFixed(2) + "万"
            }
            return num.toString()
        }
        
        // 处理市值(单位:亿元)
        if (num >= 100000000) {
            return (num / 100000000).toFixed(2) + "万亿元"
        } else if (num >= 10000) {
            return (num / 10000).toFixed(2) + "亿元"
        }
        return num.toString().toFixed(2) + "万元"
    }
}