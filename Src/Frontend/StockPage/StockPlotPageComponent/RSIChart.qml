import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3

Item {
    id: root
    property var historyData: []
    property var visibleData: []
    property int startIndex: 0
    property int visibleDays: 50
    property int rsiPeriod: 14

    // 添加计算好的RSI数据属性
    property var rsiData: []
    property var visibleRsi: []

    property color textColor: palette.text
    property color gridColor: palette.mid
    property color rsiLineColor: "#3498db"
    property color overboughtColor: "red"
    property color oversoldColor: "green"

    // 历史数据变化时重新计算RSI
    onHistoryDataChanged: {
        if (historyData.length > 0) {
            var closePrices = historyData.map(item => item.收盘价)
            rsiData = StockCalculate.calculate_rsi(closePrices, rsiPeriod)
            updateVisibleRsi()
        }
    }

    // 起始索引或可见天数变化时更新可见部分
    onStartIndexChanged: updateVisibleRsi()
    onVisibleDaysChanged: updateVisibleRsi()

    function updateVisibleRsi() {
        if (rsiData.length === 0) return
        var end = Math.min(startIndex + visibleDays, rsiData.length)
        visibleRsi = rsiData.slice(startIndex, end)
        canvas.requestPaint()
    }

    Canvas {
        id: canvas
        anchors.fill: parent

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            ctx.fillStyle = palette.base
            ctx.fillRect(0, 0, width, height)

            if (visibleRsi.length === 0) return

            // 绘制网格线
            ctx.strokeStyle = Qt.lighter(gridColor, 1.3)
            ctx.lineWidth = 0.5
            for (var i = 0; i <= 5; i++) {
                var y = i * height / 5
                ctx.beginPath()
                ctx.moveTo(0, y)
                ctx.lineTo(width, y)
                ctx.stroke()
            }

            // 绘制超买超卖线
            ctx.strokeStyle = overboughtColor
            ctx.beginPath()
            ctx.moveTo(0, height * 0.2) 
            ctx.lineTo(width, height * 0.2)
            ctx.stroke()

            ctx.strokeStyle = oversoldColor
            ctx.beginPath()
            ctx.moveTo(0, height * 0.8) 
            ctx.lineTo(width, height * 0.8)
            ctx.stroke()

            // 绘制RSI线
            var barWidth = width / visibleDays * 0.7
            var barSpacing = width / visibleDays * 0.3

            ctx.strokeStyle = rsiLineColor
            ctx.lineWidth = 1.5
            ctx.beginPath()
            var firstPoint = true
            for (var j = 0; j < visibleRsi.length; j++) {
                var rsiItem = visibleRsi[j]
                var rsiX = (j + 0.5) * (barWidth + barSpacing)
                var rsiY = height * (1 - rsiItem / 100)
                
                if (firstPoint) {
                    ctx.moveTo(rsiX, rsiY)
                    firstPoint = false
                } else {
                    ctx.lineTo(rsiX, rsiY)
                }
            }
            ctx.stroke()
        }
    }
    onVisibleDataChanged: canvas.requestPaint()
}