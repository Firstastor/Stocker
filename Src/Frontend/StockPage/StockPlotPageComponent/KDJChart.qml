import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3

Item {
    id: root
    property var historyData: []
    property var visibleData: []
    property int startIndex: 0
    property int visibleDays: 50

    // 添加计算好的KDJ数据属性
    property var kdjData: []
    property var visibleKdj: []

    property color textColor: palette.text
    property color gridColor: "#e0e0e0"
    property color kLineColor: "#3498db"
    property color dLineColor: "#f39c12"
    property color jLineColor: "#9b59b6"

    // 历史数据变化时重新计算KDJ
    onHistoryDataChanged: {
        if (historyData.length > 0) {
            var highPrices = historyData.map(item => item.最高价)
            var lowPrices = historyData.map(item => item.最低价)
            var closePrices = historyData.map(item => item.收盘价)
            kdjData = StockCalculate.calculate_kdj(highPrices, lowPrices, closePrices, 9, 3, 3)
            updateVisibleKdj()
        }
    }

    // 起始索引或可见天数变化时更新可见部分
    onStartIndexChanged: updateVisibleKdj()
    onVisibleDaysChanged: updateVisibleKdj()

    function updateVisibleKdj() {
        if (kdjData.length === 0) return
        var end = Math.min(startIndex + visibleDays, kdjData.length)
        visibleKdj = kdjData.slice(startIndex, end)
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

            if (visibleKdj.length === 0) return

            // KDJ范围固定为0-100
            var min = 0
            var max = 100
            var range = max - min

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
            ctx.strokeStyle = "red"
            ctx.beginPath()
            ctx.moveTo(0, height * 0.2) // 80%线(超买)
            ctx.lineTo(width, height * 0.2)
            ctx.stroke()

            ctx.strokeStyle = "green"
            ctx.beginPath()
            ctx.moveTo(0, height * 0.8) // 20%线(超卖)
            ctx.lineTo(width, height * 0.8)
            ctx.stroke()

            // 绘制KDJ线
            var barWidth = width / visibleDays * 0.7
            var barSpacing = width / visibleDays * 0.3

            // 绘制K线
            ctx.strokeStyle = kLineColor
            ctx.lineWidth = 1.5
            ctx.beginPath()
            var firstPoint = true
            for (var j = 0; j < visibleKdj.length; j++) {
                var kdjItem = visibleKdj[j]
                var kX = (j + 0.5) * (barWidth + barSpacing)
                var kY = height * (1 - kdjItem.k / 100)
                
                if (firstPoint) {
                    ctx.moveTo(kX, kY)
                    firstPoint = false
                } else {
                    ctx.lineTo(kX, kY)
                }
            }
            ctx.stroke()

            // 绘制D线
            ctx.strokeStyle = dLineColor
            ctx.lineWidth = 1.5
            ctx.beginPath()
            firstPoint = true
            for (var k = 0; k < visibleKdj.length; k++) {
                var kdjItemD = visibleKdj[k]
                var dX = (k + 0.5) * (barWidth + barSpacing)
                var dY = height * (1 - kdjItemD.d / 100)
                
                if (firstPoint) {
                    ctx.moveTo(dX, dY)
                    firstPoint = false
                } else {
                    ctx.lineTo(dX, dY)
                }
            }
            ctx.stroke()

            // 绘制J线
            ctx.strokeStyle = jLineColor
            ctx.lineWidth = 1.5
            ctx.beginPath()
            firstPoint = true
            for (var l = 0; l < visibleKdj.length; l++) {
                var kdjItemJ = visibleKdj[l]
                var jX = (l + 0.5) * (barWidth + barSpacing)
                var jY = height * (1 - kdjItemJ.j / 100)
                
                if (firstPoint) {
                    ctx.moveTo(jX, jY)
                    firstPoint = false
                } else {
                    ctx.lineTo(jX, jY)
                }
            }
            ctx.stroke()
        }
    }
    onVisibleDataChanged: canvas.requestPaint()
}