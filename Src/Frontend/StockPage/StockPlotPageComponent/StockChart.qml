import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3
import QtQuick.Dialogs
import QtQuick.Layouts 

Item {
    id: root

    // Data properties
    property var historyData: []
    property var visibleData: []
    property int startIndex: 0
    property int visibleDays: 50
    property int minVisibleDays: 10
    property int maxVisibleDays: 200
    property real zoomFactor: 1.0

    // Color properties
    property color upColor: "red"   
    property color downColor: "Green" 
    property color textColor: palette.text
    property color gridColor: palette.mid
    property color crosshairColor: "#808080"

    // Volume properties
    property bool showVolume: true
    property real volumeOpacity: 0.6
    property real maxVolume: 1

    // SMA properties
    property var maSettings: []
    property bool showMaSettingsDialog: false

    // Bollinger Bands properties
    property bool showBollinger: false
    property int bollingerPeriod: 20
    property real bollingerMultiplier: 2.0
    property color upperBandColor: "#e74c3c"
    property color lowerBandColor: "#2ecc71"
    property color middleBandColor: "#3498db"

    // 技术指标属性
    property string currentIndicator: "无"
    property bool showMacd: false
    property bool showRsi: false
    property bool showKdj: false

    // Interaction properties
    property bool isDragging: false
    property bool showCrosshair: false
    property point crosshairPos: Qt.point(0, 0)
    property var hoveredItem: null

    // Cached calculations
    property var bollingerBandsData: []
    property var closePrices: []

    onHistoryDataChanged: {
        closePrices = historyData.map(item => item.收盘价)
        if (showBollinger) {
            calculateBollingerBands()
        }
        for (var i = 0; i < maSettings.length; i++) {
            var setting = maSettings[i]
            setting.maData = calculateMA(setting.period, setting.type)
        }
        updateVisibleData(Math.max(0, historyData.length - visibleDays))
    }

    function calculateBollingerBands() {
        if (historyData.length < bollingerPeriod) {
            bollingerBandsData = []
            return
        }
        bollingerBandsData = StockCalculate.calculate_bollinger_bands(
            closePrices, 
            bollingerPeriod, 
            bollingerMultiplier, 
            bollingerMultiplier
        )
    }

    function calculateMA(period, type) {
        var maValues = type === "EMA" 
            ? StockCalculate.calculate_ema(closePrices, period)
            : StockCalculate.calculate_sma(closePrices, period)
        if (maValues.length === 0) return []
        
        var visibleMA = []
        for (var i = 0; i < maValues.length; i++) {
            if (i >= startIndex && i < startIndex + visibleDays) {
                visibleMA.push({
                    value: maValues[i],
                    index: i
                })
            }
        }
        return visibleMA
    }

    function updateVisibleData(newStartIndex) {
        startIndex = Math.max(0, Math.min(newStartIndex, historyData.length - visibleDays))
        visibleData = historyData.slice(startIndex, startIndex + visibleDays)
        calculateVisiblePriceRange()
        calculateMaxVolume()
        mainCanvas.requestPaint()
    }

    function calculateVisiblePriceRange() {
        if (visibleData.length === 0) return
        
        var min = visibleData[0].最低价
        var max = visibleData[0].最高价
        
        if (showBollinger && bollingerBandsData.length > 0) {
            for (var i = 0; i < visibleData.length; i++) {
                var dataIndex = startIndex + i
                var bbItem = bollingerBandsData[dataIndex]
                if (!bbItem) continue
                
                min = Math.min(min, bbItem.lower)
                max = Math.max(max, bbItem.upper)
            }
        }
        
        for (var i = 1; i < visibleData.length; i++) {
            min = Math.min(min, visibleData[i].最低价)
            max = Math.max(max, visibleData[i].最高价)
        }
        
        mainCanvas.minPrice = min - (max - min) * 0.1 
        mainCanvas.maxPrice = max + (max - min) * 0.1
        mainCanvas.priceRange = mainCanvas.maxPrice - mainCanvas.minPrice
    }

    function calculateMaxVolume() {
        if (visibleData.length === 0) {
            maxVolume = 1
            return
        }
        
        var maxVol = visibleData[0].成交量
        for (var i = 1; i < visibleData.length; i++) {
            maxVol = Math.max(maxVol, visibleData[i].成交量)
        }
        maxVolume = maxVol
    }

    onShowBollingerChanged: {
        if (showBollinger && bollingerBandsData.length === 0) {
            calculateBollingerBands()
        }
        mainCanvas.requestPaint()
    }

    onMaSettingsChanged: {
        // Update MA calculations for any new or changed settings
        for (var i = 0; i < maSettings.length; i++) {
            var setting = maSettings[i]
            if (!setting.maData || setting.maData.length === 0) {
                setting.maData = calculateMA(setting.period, setting.type)
            }
        }
        mainCanvas.requestPaint()
    }

    Canvas {
        id: mainCanvas
        anchors {
            left: priceScale.right
            right: volumeScale.left
            top: parent.top
            bottom: dateScale.top
        }
        
        property real minPrice: 0
        property real maxPrice: 0
        property real priceRange: 0
        property point dragStartPoint: Qt.point(0, 0)
        property int dragStartIndex: 0
        
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()

            // 绘制背景
            ctx.fillStyle = palette.base
            ctx.fillRect(0, 0, width, height)

            // 计算基本尺寸
            var chartWidth = width
            var barWidth = chartWidth / visibleData.length * 0.7  // K线宽度占70%
            var barSpacing = chartWidth / visibleData.length * 0.3  // 间距30%

            // 防止除零错误
            var priceRange = Math.max(0.0001, maxPrice - minPrice)
            var volumeRange = Math.max(1, maxVolume)

            // 成交量柱子高度最大只占K线区底部20%
            var volumeBarMaxHeight = height * 0.4
            var volumeBarBaseY = height

            // 绘制网格线（全高）
            function drawGridLines() {
                ctx.strokeStyle = Qt.lighter(gridColor, 1.3)
                ctx.lineWidth = 0.5
                
                // 水平网格线
                var hSteps = 5
                for (var i = 0; i <= hSteps; i++) {
                    var y = i * height / hSteps
                    ctx.beginPath()
                    ctx.moveTo(0, y)
                    ctx.lineTo(width, y)
                    ctx.stroke()
                }

                // 垂直网格线
                var vSteps = Math.min(12, Math.max(5, Math.floor(chartWidth / 120)))
                for (var j = 0; j <= vSteps; j++) {
                    var x = j * chartWidth / vSteps
                    ctx.beginPath()
                    ctx.moveTo(x, 0)
                    ctx.lineTo(x, height)
                    ctx.stroke()
                }
            }

            function drawBollingerBands() {
                var drawBandLine = function(color, getY) {
                    ctx.strokeStyle = color
                    ctx.beginPath()
                    var firstPoint = true
                    for (var j = 0; j < visibleData.length; j++) {
                        var idx = startIndex + j
                        var item = bollingerBandsData[idx]
                        if (!item) continue
                        
                        var x = (j + 0.5) * (barWidth + barSpacing)
                        var y = height - ((getY(item) - minPrice) / priceRange * height)
                        
                        if (firstPoint) {
                            ctx.moveTo(x, y)
                            firstPoint = false
                        } else {
                            ctx.lineTo(x, y)
                        }
                    }
                    ctx.stroke()
                }
                
                drawBandLine(upperBandColor, (item) => item.upper)
                drawBandLine(middleBandColor, (item) => item.middle)
                drawBandLine(lowerBandColor, (item) => item.lower)
            }

            // 绘制K线
            function drawKlines() {
                for (var k = 0; k < visibleData.length; k++) {
                    var item = visibleData[k]
                    var x = k * (barWidth + barSpacing) + barSpacing/2

                    // 计算价格坐标（全高）
                    var openY = height - ((item.开盘价 - minPrice) / priceRange * height)
                    var closeY = height - ((item.收盘价 - minPrice) / priceRange * height)
                    var highY = height - ((item.最高价 - minPrice) / priceRange * height)
                    var lowY = height - ((item.最低价 - minPrice) / priceRange * height)

                    var isUp = item.收盘价 >= item.开盘价
                    ctx.fillStyle = isUp ? upColor : downColor
                    ctx.strokeStyle = isUp ? upColor : downColor

                    // 绘制影线
                    ctx.beginPath()
                    ctx.moveTo(x + barWidth/2, highY)
                    ctx.lineTo(x + barWidth/2, lowY)
                    ctx.stroke()

                    // 绘制实体
                    var bodyTop = Math.min(openY, closeY)
                    var bodyHeight = Math.max(1, Math.abs(openY - closeY)) 
                    ctx.fillRect(x, bodyTop, barWidth, bodyHeight)
                }
            }

            // 绘制成交量
            function drawVolumes() {
                for (var k = 0; k < visibleData.length; k++) {
                    var item = visibleData[k]
                    var x = k * (barWidth + barSpacing) + barSpacing/2
                    var isUp = item.收盘价 >= item.开盘价

                    var volHeight = (item.成交量 / maxVolume) * volumeBarMaxHeight
                    var volY = height - volHeight

                    ctx.fillStyle = isUp ? Qt.rgba(upColor.r, upColor.g, upColor.b, volumeOpacity)
                                        : Qt.rgba(downColor.r, downColor.g, downColor.b, volumeOpacity)
                    ctx.fillRect(x, volY, barWidth, volHeight)
                }
            }

            function drawMALines() {
                for (var l = 0; l < maSettings.length; l++) {
                    var setting = maSettings[l]
                    if (!setting.visible || !setting.maData) continue
                    
                    ctx.strokeStyle = setting.color
                    ctx.lineWidth = 1.5
                    ctx.beginPath()
                    
                    var firstPoint = true
                    for (var m = 0; m < setting.maData.length; m++) {
                        var maItem = setting.maData[m]
                        var relativeIndex = maItem.index - startIndex
                        if (relativeIndex < 0 || relativeIndex >= visibleData.length) continue
                        
                        var maX = (relativeIndex + 0.5) * (barWidth + barSpacing)
                        var maY = height - ((maItem.value - minPrice) / priceRange * height)
                        
                        if (firstPoint) {
                            ctx.moveTo(maX, maY)
                            firstPoint = false
                        } else {
                            ctx.lineTo(maX, maY)
                        }
                    }
                    ctx.stroke()
                }
            }

            function drawCrosshair() {
                if (!showCrosshair || !hoveredItem) return

                ctx.strokeStyle = crosshairColor
                ctx.lineWidth = 1
                ctx.setLineDash([3, 3])
                
                // 水平线
                ctx.beginPath()
                ctx.moveTo(0, crosshairPos.y)
                ctx.lineTo(width, crosshairPos.y)
                ctx.stroke()
                
                // 垂直线
                ctx.beginPath()
                ctx.moveTo(crosshairPos.x, 0)
                ctx.lineTo(crosshairPos.x, height)
                ctx.stroke()
                
                ctx.setLineDash([])

                // 计算价格和日期
                var price = minPrice + (height - crosshairPos.y) / height * priceRange
                var date = new Date(hoveredItem.日期)
                var volume = hoveredItem.成交量

                // 价格标签
                var priceLabelWidth = 60
                var priceLabelX
                priceLabelX = 5
                ctx.textAlign = "left"
                ctx.fillStyle = Qt.rgba(palette.base.r, palette.base.g, palette.base.b, 0.8)
                ctx.strokeStyle = textColor
                ctx.fillRect(priceLabelX, crosshairPos.y - 10, priceLabelWidth, 20)
                ctx.strokeRect(priceLabelX, crosshairPos.y - 10, priceLabelWidth, 20)
                ctx.fillStyle = textColor
                ctx.font = "10px sans-serif"
                ctx.fillText(price.toFixed(2), priceLabelX + (ctx.textAlign === "left" ? 5 : priceLabelWidth - 5), crosshairPos.y + 5)

                // 日期标签
                ctx.textAlign = "center"
                ctx.fillStyle = Qt.rgba(palette.base.r, palette.base.g, palette.base.b, 0.8)
                ctx.strokeStyle = textColor
                ctx.fillRect(crosshairPos.x - 30, height - 20, 60, 20)
                ctx.strokeRect(crosshairPos.x - 30, height - 20, 60, 20)
                ctx.fillStyle = textColor
                ctx.fillText(Qt.formatDate(date, "yyyy/MM/dd"), crosshairPos.x, height - 5)

                // 成交量标签
                ctx.textAlign = "right"
                ctx.fillStyle = Qt.rgba(palette.base.r, palette.base.g, palette.base.b, 0.8)
                ctx.strokeStyle = textColor
                ctx.fillRect(width - 70, crosshairPos.y - 10, 70, 20)
                ctx.strokeRect(width - 70, crosshairPos.y - 10, 70, 20)
                ctx.fillStyle = textColor
                var volText = volume >= 1000000 ? (volume / 1000000).toFixed(2) + "M" : 
                            volume >= 1000 ? (volume / 1000).toFixed(1) + "K" : volume
                ctx.fillText(volText, width - 5, crosshairPos.y + 5)
            }

            // 执行绘制顺序
            drawGridLines()
            drawKlines()
            if (showBollinger) drawBollingerBands()
            drawVolumes()
            drawMALines()
            drawCrosshair()
        }

        WheelHandler {
            onWheel: function(event) {
                var centerIndex = startIndex + visibleDays * (event.x / width)
                var zoom = 1.0 + event.angleDelta.y * 0.001
                zoomFactor *= zoom
                zoomFactor = Math.max(0.5, Math.min(zoomFactor, 4.0))
                visibleDays = Math.max(minVisibleDays, Math.min(maxVisibleDays, 
                                      Math.round(maxVisibleDays / zoomFactor)))
                var newStartIndex = Math.round(centerIndex - visibleDays * (event.x / width))
                updateVisibleData(newStartIndex)
                event.accepted = true
            }
        }

        MouseArea {
            id: chartMouseArea
            anchors.fill: parent
            preventStealing: true
            hoverEnabled: true
            
            property real dragSensitivity: 0.5
            property real snapThreshold: 20
            property bool enablePriceSnap: true
            property point dragStartPoint: Qt.point(0, 0)
            property int dragStartIndex: 0
            
            onPressed: function(mouse) {
                dragStartIndex = startIndex
                dragStartPoint = Qt.point(mouse.x, mouse.y)
                isDragging = true
                showCrosshair = true
                updateCrosshairPosition(mouse.x, mouse.y)
            }
            
            onReleased: {
                isDragging = false
            }
            
            onPositionChanged: function(mouse) {
                handleDrag(mouse)
                updateCrosshairPosition(mouse.x, mouse.y)
            }
            
            onExited: {
                showCrosshair = false
                mainCanvas.requestPaint()
            }
            
            function handleDrag(mouse) {
                if (isDragging) {
                    var deltaX = mouse.x - dragStartPoint.x
                    var pixelsPerDay = width / visibleDays
                    var daysToMove = Math.round(deltaX * dragSensitivity / pixelsPerDay)
                    
                    if (daysToMove !== 0) {
                        var newStartIndex = Math.max(0, Math.min(
                            historyData.length - visibleDays, 
                            dragStartIndex - daysToMove
                        ))
                        updateVisibleData(newStartIndex)
                        dragStartPoint = Qt.point(mouse.x, mouse.y)
                        dragStartIndex = newStartIndex
                    }
                }
            }
            
            function updateCrosshairPosition(x, y) {
                if (showCrosshair) {
                    var snappedPos = enablePriceSnap ? snapToNearestPrice(x, y) : snapToNearestDay(x, y)
                    crosshairPos = snappedPos
                    updateHoveredItem(snappedPos.x)
                    mainCanvas.requestPaint()
                }
            }
            
            function snapToNearestDay(x) {
                if (visibleData.length === 0) return Qt.point(x, y)
                
                var dayWidth = width / visibleDays
                var dayIndex = Math.round(x / dayWidth)
                dayIndex = Math.min(Math.max(dayIndex, 0), visibleDays - 1)
                return Qt.point(dayIndex * dayWidth + dayWidth/2, y)
            }
            
            function snapToNearestPrice(x, y) {
                if (visibleData.length === 0) return Qt.point(x, y)
                
                var snappedX = snapToNearestDay(x).x
                var dayWidth = width / visibleDays
                var dayIndex = Math.round(x / dayWidth)
                var item = visibleData[Math.min(dayIndex, visibleData.length - 1)]
                if (!item) return Qt.point(snappedX, y)
                
                var minPrice = mainCanvas.minPrice - mainCanvas.priceRange * 0.1
                var priceRange = mainCanvas.priceRange * 1.2
                var pricePoints = [
                    {value: item.开盘价, name: "开"},
                    {value: item.收盘价, name: "收"},
                    {value: item.最高价, name: "高"},
                    {value: item.最低价, name: "低"}
                ]
                
                var closestY = y
                var minDistance = Infinity
                
                for (var i = 0; i < pricePoints.length; i++) {
                    var priceY = height - ((pricePoints[i].value - minPrice) / priceRange * height)
                    var distance = Math.abs(y - priceY)
                    
                    if (distance < minDistance && distance <= snapThreshold) {
                        minDistance = distance
                        closestY = priceY
                    }
                }
                
                return Qt.point(snappedX, closestY)
            }
            
            function updateHoveredItem(x) {
                if (visibleData.length === 0) return
                
                var dayWidth = width / visibleDays
                var dayIndex = Math.floor(x / dayWidth)
                dayIndex = Math.min(Math.max(dayIndex, 0), visibleData.length - 1)
                hoveredItem = visibleData[dayIndex]
            }
        }
    }

    Column {
        id: priceScale
        anchors {
            left: parent.left
            top: parent.top
            bottom: dateScale.top
        }
        width: 30
        
        Repeater {
            model: 3
            Text {
                width: parent.width
                height: parent.height / 3
                text: (mainCanvas.minPrice + (2-index) * mainCanvas.priceRange/2).toFixed(2)
                color: textColor
                font.pixelSize: 10
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: index === 0 ? Text.AlignTop : 
                              index === 2 ? Text.AlignBottom : Text.AlignVCenter
            }
        }
    }


    Column {
        id: volumeScale
        anchors {
            right: parent.right
            top: parent.top
            bottom: dateScale.top
        }
        width: 40
        
        Repeater {
            model: 3
            Text {
                width: parent.width
                height: parent.height / 3
                text: {
                    if (index === 0) return (maxVolume / 1000000).toFixed(1) + "M"
                    else if (index === 1) return (maxVolume / 2000000).toFixed(1) + "M"
                    else return "0"
                }
                color: textColor
                font.pixelSize: 10
                horizontalAlignment: Text.AlignRight
                verticalAlignment: index === 0 ? Text.AlignTop : 
                              index === 2 ? Text.AlignBottom : Text.AlignVCenter
            }
        }
    }

    Row {
        id: dateScale
        anchors {
            left: parent.left
            right: parent.right
            bottom: parent.bottom
        }
        height: 20

        Repeater {
            model: 5
            delegate: Text {
                width: parent.width / 5
                height: parent.height
                color: textColor
                font.pixelSize: 10
                horizontalAlignment: Text.AlignHCenter
                text: {
                    if (visibleData.length === 0) return ""
                    var idx = Math.round(index * (visibleData.length - 1) / 4)
                    return Qt.formatDate(new Date(visibleData[idx].日期), "yyyy/MM/dd")
                }
            }
        }
    }

    Button {
        id: addMaButton
        width: 24
        height: 24
        z: 1
        anchors.bottom: parent.top
        anchors.right: parent.right
        anchors.margins: 5
        text: "+"
        font.pixelSize: 16
        onClicked: showMaSettingsDialog = true
        contentItem: Text {
            text: addMaButton.text
            font: addMaButton.font
            color: textColor
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }

    // 技术指标按钮组
    Row {
        id: indicatorButtons
        spacing: 5
        anchors.bottom: parent.top
        anchors.right: addMaButton.left
        anchors.margins: 5
        z: 1
        
        Button {
            id: macdBtn
            width: 60
            height: 24
            text: showMacd ? "隐藏MACD" : "MACD"
            onClicked: {
                currentIndicator = "MACD"
                showMacd = !showMacd
                showRsi = false
                showKdj = false
            }
            contentItem: Text {
                text: macdBtn.text
                font: macdBtn.font
                color: textColor
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }
        
        Button {
            id: rsiBtn
            width: 60
            height: 24
            text: showRsi ? "隐藏RSI" : "RSI"
            onClicked: {
                currentIndicator = "RSI"
                showRsi = !showRsi
                showMacd = false
                showKdj = false
            }
            contentItem: Text {
                text: rsiBtn.text
                font: rsiBtn.font
                color: textColor
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }
        
        Button {
            id: kdjBtn
            width: 60
            height: 24
            text: showKdj ? "隐藏KDJ" : "KDJ"
            onClicked: {
                currentIndicator = "KDJ"
                showKdj = !showKdj
                showMacd = false
                showRsi = false
            }
            contentItem: Text {
                text: kdjBtn.text
                font: kdjBtn.font
                color: textColor
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }
        
        Button {
            id: bollingerToggle
            width: 80
            height: 24
            text: showBollinger ? "隐藏布林带" : "布林带"
            onClicked: showBollinger = !showBollinger
            
            contentItem: Text {
                text: bollingerToggle.text
                font: bollingerToggle.font
                color: textColor
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }
    }

    Popup {
        id: maSettingsDialog
        visible: showMaSettingsDialog
        width: 220
        height: 220
        x: parent.width - width - 5
        y: addMaButton.y + addMaButton.height + 5
        modal: true
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        
        background: Rectangle {
            color: palette.base
            border.color: gridColor
            border.width: 1
            radius: 4
        }
        
        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 10
            spacing: 10
            
            Label {
                text: "添加MA指标"
                font.bold: true
                Layout.alignment: Qt.AlignHCenter
            }
            
            RowLayout {
                Label { text: "类型:" }
                ComboBox {
                    id: maType
                    Layout.fillWidth: true
                    model: ["SMA", "EMA"]
                    currentIndex: 0
                }
            }
            
            RowLayout {
                Label { text: "周期:" }
                TextField {
                    id: maPeriod
                    Layout.fillWidth: true
                    text: "20"
                    inputMethodHints: Qt.ImhDigitsOnly
                    onEditingFinished: {
                        var value = parseInt(text)
                        if (value < 5) text = "5"
                        else if (value > 60) text = "60"
                    }
                }
            }
            
            RowLayout {
                Label { text: "颜色:" }
                Button {
                    id: colorButton
                    text: "选择颜色"
                    onClicked: colorDialog.open()
                    
                    Rectangle {
                        width: 16
                        height: 16
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.rightMargin: 5
                        color: colorDialog.selectedColor
                        border.color: textColor
                        border.width: 1
                    }
                }
            }
            
            RowLayout {
                spacing: 10
                Button {
                    text: "添加"
                    Layout.fillWidth: true
                    onClicked: {
                        var period = parseInt(maPeriod.text)
                        var type = maType.currentText
                        var color = colorDialog.selectedColor.toString()
                        var existingIndex = maSettings.findIndex(function(item) {
                            return item.period === period && item.type === type
                        })
                        if (existingIndex >= 0) {
                            maSettings[existingIndex].color = color
                        } else {
                            var newMa = {
                                period: period,
                                type: type,
                                color: color,
                                visible: true
                            }
                            maSettings.push(newMa)
                        }
                        maSettings = maSettings.slice()
                        showMaSettingsDialog = false
                        updateVisibleData(startIndex)
                    }
                }
                
                Button {
                    text: "取消"
                    Layout.fillWidth: true
                    onClicked: showMaSettingsDialog = false
                }
            }
        }
    }

    ColorDialog {
        id: colorDialog
        title: "选择MA颜色"
        selectedColor: "#3498db"
    }

    Row {
        id: legend
        spacing: 5
        anchors.bottom: parent.top
        anchors.right: indicatorButtons.left
        anchors.margins: 10
        visible: maSettings.length > 0
        layoutDirection: Qt.RightToLeft
        
        Repeater {
            model: maSettings
            
            delegate: Rectangle {
                color: "transparent"
                width: childrenRect.width
                height: childrenRect.height
                
                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        maSettings.splice(index, 1)
                        maSettings = maSettings.slice()
                    }
                }
                
                Row {
                    spacing: 5
                    Text {
                        text: modelData.type + modelData.period
                        color: textColor
                        font.pixelSize: 10
                    }
                    
                    Rectangle {
                        width: 15
                        height: 2
                        color: modelData.color
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }
        }
    }
}