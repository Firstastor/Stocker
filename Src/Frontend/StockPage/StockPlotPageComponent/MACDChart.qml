import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3

Item {
    id: root
    property var historyData: []
    property var visibleData: []
    property int startIndex: 0
    property int visibleDays: 50

    // 添加计算好的MACD数据属性
    property var macdData: []
    property var visibleMacd: []

    property color textColor: palette.text
    property color gridColor: palette.mid
    property color positiveColor: "red"
    property color negativeColor: "green"
    property color lineColor: "#3498db"
    property color signalLineColor: "#f39c12"

    // 历史数据变化时重新计算MACD
    onHistoryDataChanged: {
        if (historyData.length > 0) {
            var closePrices = historyData.map(item => item.收盘价)
            macdData = StockCalculate.calculate_macd(closePrices, 12, 26, 9)
            updateVisibleMacd()
        }
    }

    // 起始索引或可见天数变化时更新可见部分
    onStartIndexChanged: updateVisibleMacd()
    onVisibleDaysChanged: updateVisibleMacd()

    function updateVisibleMacd() {
        if (macdData.length === 0) return
        var end = Math.min(startIndex + visibleDays, macdData.length)
        visibleMacd = macdData.slice(startIndex, end)
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
            
            if (visibleMacd.length === 0) return

            // 计算显示范围
            var min = 0, max = 0
            visibleMacd.forEach(function(item) {
                min = Math.min(min, item.macd, item.signal, item.histogram)
                max = Math.max(max, item.macd, item.signal, item.histogram)
            })
            var range = Math.max(0.0001, max - min)

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

            // 绘制零线
            ctx.strokeStyle = textColor
            ctx.lineWidth = 1
            var zeroY = height * (1 - (-min) / range)
            ctx.beginPath()
            ctx.moveTo(0, zeroY)
            ctx.lineTo(width, zeroY)
            ctx.stroke()

            // 绘制柱状图
            var barWidth = width / visibleDays * 0.7
            var barSpacing = width / visibleDays * 0.3
            for (var j = 0; j < visibleMacd.length; j++) {
                var item = visibleMacd[j]
                var x = j * (barWidth + barSpacing) + barSpacing/2
                var histY = height * (1 - (item.histogram - min) / range)
                var histHeight = Math.abs(item.histogram / range * height)

                ctx.fillStyle = item.histogram >= 0 ? positiveColor : negativeColor
                ctx.fillRect(x, histY, barWidth, histHeight)
            }

            // 绘制MACD线
            ctx.strokeStyle = lineColor
            ctx.lineWidth = 1.5
            ctx.beginPath()
            var firstPoint = true
            for (var k = 0; k < visibleMacd.length; k++) {
                var macdItem = visibleMacd[k]
                var macdX = (k + 0.5) * (barWidth + barSpacing)
                var macdY = height * (1 - (macdItem.macd - min) / range)
                
                if (firstPoint) {
                    ctx.moveTo(macdX, macdY)
                    firstPoint = false
                } else {
                    ctx.lineTo(macdX, macdY)
                }
            }
            ctx.stroke()

            // 绘制信号线
            ctx.strokeStyle = signalLineColor
            ctx.lineWidth = 1.5
            ctx.beginPath()
            firstPoint = true
            for (var l = 0; l < visibleMacd.length; l++) {
                var signalItem = visibleMacd[l]
                var signalX = (l + 0.5) * (barWidth + barSpacing)
                var signalY = height * (1 - (signalItem.signal - min) / range)
                
                if (firstPoint) {
                    ctx.moveTo(signalX, signalY)
                    firstPoint = false
                } else {
                    ctx.lineTo(signalX, signalY)
                }
            }
            ctx.stroke()
        }
    }

    onVisibleDataChanged: canvas.requestPaint()
}