import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3
import QtQuick.Dialogs
import QtQuick.Layouts

Item {
    id: root
    property var historyData: []
    property color upColor: "red"
    property color downColor: "green"
    property color bgColor: palette.window
    property color textColor: palette.text
    property color gridColor: "#e0e0e0"
    property color crosshairColor: "#808080"
    property int visibleDays: 100
    property int minVisibleDays: 10
    property int maxVisibleDays: 300
    property real zoomFactor: 1.0
    property int startIndex: 0
    property var visibleData: []
    property bool isDragging: false
    property bool showCrosshair: false
    property point crosshairPos: Qt.point(0, 0)
    property var hoveredItem: null
    
    // SMA相关属性
    property var smaSettings: [
        { period: 20, color: "blue", visible: true },
        { period: 5, color: "orange" , visible: true }
    ]
    property bool showSmaSettingsDialog: false

    onHistoryDataChanged: updateVisibleData(Math.max(0, historyData.length - visibleDays))
    onSmaSettingsChanged: canvas.requestPaint() 

    function updateVisibleData(newStartIndex) {
        startIndex = Math.max(0, Math.min(newStartIndex, historyData.length - visibleDays))
        visibleData = historyData.slice(startIndex, startIndex + visibleDays)
        calculateVisiblePriceRange()
        canvas.requestPaint()
        gridCanvas.requestPaint()
    }

    function calculateVisiblePriceRange() {
        if (visibleData.length === 0) return
        
        var min = visibleData[0].最低价
        var max = visibleData[0].最高价
        
        for (var i = 1; i < visibleData.length; i++) {
            min = Math.min(min, visibleData[i].最低价)
            max = Math.max(max, visibleData[i].最高价)
        }
        
        canvas.minPrice = min - (max - min) * 0.1 
        canvas.maxPrice = max + (max - min) * 0.1
        canvas.priceRange = canvas.maxPrice - canvas.minPrice
    }

    function calculateSMA(data, period) {
        if (data.length < period) return []
        
        // 计算完整数据的SMA
        var fullSMA = []
        for (var i = period - 1; i < data.length; i++) {
            var sum = 0
            for (var j = 0; j < period; j++) {
                sum += data[i - j].收盘价
            }
            fullSMA.push({value: sum / period, index: i})
        }
        
        // 只返回当前可见区域的SMA点
        var visibleSMA = []
        for (var k = 0; k < fullSMA.length; k++) {
            var smaItem = fullSMA[k]
            if (smaItem.index >= startIndex && smaItem.index < startIndex + visibleDays) {
                visibleSMA.push(smaItem)
            }
        }
        return visibleSMA
    }

    Canvas {
        id: gridCanvas
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            
            // 绘制背景渐变
            var gradient = ctx.createLinearGradient(0, 0, 0, height)
            gradient.addColorStop(0, Qt.lighter(root.bgColor, 1.1))
            gradient.addColorStop(1, Qt.darker(root.bgColor, 1.1))
            ctx.fillStyle = gradient
            ctx.fillRect(0, 0, width, height)
            
            // 设置网格样式
            ctx.strokeStyle = Qt.lighter(root.gridColor, 1.3)
            ctx.lineWidth = 0.5
            
            // 计算网格步数
            var hSteps = Math.min(8, Math.max(4, Math.floor(height / 80)))
            var vSteps = Math.min(12, Math.max(5, Math.floor(width / 120)))
            
            // 绘制水平网格线
            for (var i = 0; i <= hSteps; i++) {
                var y = i * height / hSteps
                ctx.beginPath()
                ctx.moveTo(0, y)
                ctx.lineTo(width, y)
                ctx.stroke()
            }
            
            // 绘制垂直网格线
            for (var j = 0; j <= vSteps; j++) {
                var x = j * width / vSteps
                ctx.beginPath()
                ctx.moveTo(x, 0)
                ctx.lineTo(x, height)
                ctx.stroke()
            }
            
            // 绘制中心基准线
            ctx.strokeStyle = Qt.darker(root.gridColor, 1.5)
            ctx.lineWidth = 1
            ctx.setLineDash([2, 2])
            
            // 水平中线
            ctx.beginPath()
            ctx.moveTo(0, height/2)
            ctx.lineTo(width, height/2)
            ctx.stroke()
            
            ctx.setLineDash([])
            
            // 绘制边框
            ctx.strokeStyle = Qt.darker(root.gridColor, 1.8)
            ctx.lineWidth = 1
            ctx.strokeRect(0, 0, width, height)
        }
    }

    // 加号按钮 - 添加SMA指标
    Button {
        id: addSmaButton
        width: 24
        height: 24
        z: 1
        anchors.bottom: parent.top
        anchors.right: parent.right
        anchors.margins: 5
        text: "+"
        contentItem: Text {
            text: addSmaButton.text
            font: addSmaButton.font
            color: palette.text
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        font.pixelSize: 16
        onClicked: showSmaSettingsDialog = true
    }

    // SMA设置对话框
    Popup {
        id: smaSettingsDialog
        visible: showSmaSettingsDialog
        width: 200
        height: 180
        x: parent.width - width - 5
        y: addSmaButton.y + addSmaButton.height + 5
        modal: true
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        
        background: Rectangle {
            color: root.bgColor
            border.color: root.gridColor
            border.width: 1
            radius: 4
        }
        
        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 10
            spacing: 10
            
            Label {
                text: "添加SMA指标"
                font.bold: true
                Layout.alignment: Qt.AlignHCenter
            }
            
            RowLayout {
                Label { text: "周期:" }
                TextField {
                    id: smaPeriod
                    Layout.fillWidth: true
                    text: "20" // 默认值
                    inputMethodHints: Qt.ImhDigitsOnly // 仅允许数字输入
                    onEditingFinished: {
                        // 确保输入值在范围内
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
                        border.color: root.textColor
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
                        var period = parseInt(smaPeriod.text);
                        var color = colorDialog.selectedColor.toString();

                        // 检查是否已存在相同周期的SMA设置
                        var existingIndex = smaSettings.findIndex(function(item) {
                            return item.period === period;
                        });

                        if (existingIndex >= 0) {
                            // 如果已存在，更新颜色
                            smaSettings[existingIndex].color = color;
                        } else {
                            // 如果不存在，新增一条SMA线
                            var newSma = {
                                period: period,
                                color: color,
                                visible: true
                            };
                            smaSettings.push(newSma);
                        }

                        smaSettings = smaSettings.slice();
                        showSmaSettingsDialog = false;
                    }
                }
                
                Button {
                    text: "取消"
                    Layout.fillWidth: true
                    onClicked: showSmaSettingsDialog = false
                }
            }
        }
    }

    // 颜色选择对话框
    ColorDialog {
        id: colorDialog
        title: "选择MA颜色"
        selectedColor: "blue"
    }

 
    // 图例区域
    Row {
        id: legend
        spacing: 5
        anchors.bottom: parent.top
        anchors.right: addSmaButton.left  
        anchors.margins: 10
        visible: smaSettings.length > 0
        layoutDirection: Qt.RightToLeft  

        Repeater {
            model: smaSettings
            
            delegate: Rectangle {
                color: "transparent"
                width: childrenRect.width 
                height: childrenRect.height 
                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        smaSettings.splice(index, 1)
                        smaSettings = smaSettings.slice()
                    }
                }

                Row {
                    spacing: 5
                    Text {
                        text: "M" + modelData.period
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
    
    Canvas {
        id: canvas
        property real minPrice: 0
        property real maxPrice: 0
        property real priceRange: 0
        property point dragStartPoint: Qt.point(0, 0)
        property int dragStartIndex: 0
        
        anchors.fill: parent
        
        WheelHandler {
            onWheel: function(event) {
                var centerIndex = startIndex + visibleDays * (event.x / width)
                var zoom = 1.0 + event.angleDelta.y * 0.001
                root.zoomFactor *= zoom
                root.zoomFactor = Math.max(0.5, Math.min(root.zoomFactor, 4.0))
                root.visibleDays = Math.max(root.minVisibleDays, Math.min(root.maxVisibleDays, 
                                          Math.round(root.maxVisibleDays / root.zoomFactor)))
                var newStartIndex = Math.round(centerIndex - root.visibleDays * (event.x / width))
                root.updateVisibleData(newStartIndex)
                event.accepted = true
            }
        }
        
        MouseArea {
            id: chartMouseArea
            anchors.fill: parent
            preventStealing: true
            hoverEnabled: true
            
            // 可配置参数
            property real dragSensitivity: 0.5  // 拖动灵敏度 (0.1-1.0)
            property real snapThreshold: 20    // 吸附阈值(像素)
            property bool enablePriceSnap: true // 是否启用价格吸附
            
            // 内部状态
            property point dragStartPoint: Qt.point(0, 0)
            property int dragStartIndex: 0
            
            onPressed: function(mouse) {
                dragStartIndex = root.startIndex
                dragStartPoint = Qt.point(mouse.x, mouse.y)
                root.isDragging = true
                root.showCrosshair = true
                updateCrosshairPosition(mouse.x, mouse.y)
            }
            
            onReleased: {
                root.isDragging = false
            }
            
            onPositionChanged: function(mouse) {
                handleDrag(mouse)
                updateCrosshairPosition(mouse.x, mouse.y)
            }
            
            onExited: {
                root.showCrosshair = false
                canvas.requestPaint()
            }
            
            // 处理拖动逻辑
            function handleDrag(mouse) {
                if (root.isDragging) {
                    var deltaX = mouse.x - dragStartPoint.x
                    var pixelsPerDay = width / root.visibleDays
                    var daysToMove = Math.round(deltaX * dragSensitivity / pixelsPerDay)
                    
                    if (daysToMove !== 0) {
                        var newStartIndex = Math.max(0, Math.min(
                            root.historyData.length - root.visibleDays, 
                            dragStartIndex - daysToMove
                        ))
                        root.updateVisibleData(newStartIndex)
                        dragStartPoint = Qt.point(mouse.x, mouse.y)
                        dragStartIndex = newStartIndex
                    }
                }
            }
            
            // 更新十字光标位置（带吸附功能）
            function updateCrosshairPosition(x, y) {
                if (root.showCrosshair) {
                    var snappedPos = enablePriceSnap ? snapToNearestPrice(x, y) : snapToNearestDay(x, y)
                    root.crosshairPos = snappedPos
                    updateHoveredItem(snappedPos.x)
                    canvas.requestPaint()
                }
            }
            
            // X轴吸附到最近的交易日
            function snapToNearestDay(x) {
                if (root.visibleData.length === 0) return Qt.point(x, y)
                
                var dayWidth = width / root.visibleDays
                var dayIndex = Math.round(x / dayWidth)
                dayIndex = Math.min(Math.max(dayIndex, 0), root.visibleDays - 1)
                return Qt.point(dayIndex * dayWidth + dayWidth/2, y)
            }
            
            // 吸附到最近的K线价格点（开盘/收盘/最高/最低）
            function snapToNearestPrice(x, y) {
                if (root.visibleData.length === 0) return Qt.point(x, y)
                
                // 先进行X轴吸附
                var snappedX = snapToNearestDay(x).x
                
                // 获取当前交易日数据
                var dayWidth = width / root.visibleDays
                var dayIndex = Math.round(x / dayWidth)
                var item = root.visibleData[Math.min(dayIndex, root.visibleData.length - 1)]
                if (!item) return Qt.point(snappedX, y)
                
                // 计算各关键价格点的Y坐标
                var minPrice = canvas.minPrice - canvas.priceRange * 0.1
                var priceRange = canvas.priceRange * 1.2
                var pricePoints = [
                    {value: item.开盘价, name: "开"},
                    {value: item.收盘价, name: "收"},
                    {value: item.最高价, name: "高"},
                    {value: item.最低价, name: "低"}
                ]
                
                // 找出最近的Y坐标
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
            
            // 更新当前悬停的K线数据
            function updateHoveredItem(x) {
                if (root.visibleData.length === 0) return
                
                var dayWidth = width / root.visibleDays
                var dayIndex = Math.floor(x / dayWidth)
                dayIndex = Math.min(Math.max(dayIndex, 0), root.visibleData.length - 1)
                root.hoveredItem = root.visibleData[dayIndex]
            }
        }
        
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            
            var minPrice = canvas.minPrice
            var maxPrice = canvas.maxPrice
            var priceRange = canvas.priceRange
            
            minPrice -= priceRange * 0.1
            maxPrice += priceRange * 0.1
            priceRange = maxPrice - minPrice
            
            var barWidth = width / root.visibleData.length * 0.7
            var barSpacing = width / root.visibleData.length * 0.3
            
            // 绘制K线
            for (var j = 0; j < root.visibleData.length; j++) {
                var item = root.visibleData[j]
                var x = j * (barWidth + barSpacing) + barSpacing/2
                
                var openY = height - ((item.开盘价 - minPrice) / priceRange * height)
                var closeY = height - ((item.收盘价 - minPrice) / priceRange * height)
                var highY = height - ((item.最高价 - minPrice) / priceRange * height)
                var lowY = height - ((item.最低价 - minPrice) / priceRange * height)
                
                var isUp = item.收盘价 >= item.开盘价
                ctx.fillStyle = isUp ? root.upColor : root.downColor
                ctx.strokeStyle = isUp ? root.upColor : root.downColor
                
                ctx.beginPath()
                ctx.moveTo(x + barWidth/2, highY)
                ctx.lineTo(x + barWidth/2, lowY)
                ctx.stroke()
                
                var bodyTop = Math.min(openY, closeY)
                var bodyHeight = Math.abs(openY - closeY)
                if (bodyHeight < 1) bodyHeight = 1
                
                ctx.fillRect(x, bodyTop, barWidth, bodyHeight)
            }
            
            // 绘制SMA指标
            for (var k = 0; k < smaSettings.length; k++) {
                var setting = smaSettings[k]
                if (!setting.visible) continue
                
                // 计算完整数据的SMA
                var fullSMA = calculateSMA(root.historyData, setting.period)
                if (fullSMA.length === 0) continue
                
                ctx.strokeStyle = setting.color
                ctx.lineWidth = 1.5
                ctx.beginPath()
                
                // 计算可见区域的SMA点
                for (var m = 0; m < fullSMA.length; m++) {
                    var smaItem = fullSMA[m]
                    // 转换为可见区域的相对位置
                    var relativeIndex = smaItem.index - startIndex
                    if (relativeIndex < 0) continue
                    
                    var smaX = (relativeIndex + 0.5) * (barWidth + barSpacing)
                    var smaY = height - ((smaItem.value - minPrice) / priceRange * height)
                    
                    if (m === 0 || (relativeIndex === 0 && m > 0)) {
                        ctx.moveTo(smaX, smaY)
                    } else {
                        ctx.lineTo(smaX, smaY)
                    }
                }
                
                ctx.stroke()
            }
            
            // 绘制十字光标
            if (root.showCrosshair) {
                ctx.strokeStyle = root.crosshairColor
                ctx.lineWidth = 1
                ctx.setLineDash([3, 3])
                
                // 水平线
                ctx.beginPath()
                ctx.moveTo(0, root.crosshairPos.y)
                ctx.lineTo(width, root.crosshairPos.y)
                ctx.stroke()
                
                // 垂直线
                ctx.beginPath()
                ctx.moveTo(root.crosshairPos.x, 0)
                ctx.lineTo(root.crosshairPos.x, height)
                ctx.stroke()
                
                ctx.setLineDash([])
                
                // 绘制价格和日期信息
                if (root.hoveredItem) {
                    var price = minPrice + (height - root.crosshairPos.y) / height * priceRange
                    var date = new Date(root.hoveredItem.日期)
                    
                    ctx.fillStyle = root.bgColor
                    ctx.strokeStyle = root.textColor
                    ctx.lineWidth = 1
                    
                    // 价格标签背景
                    ctx.fillRect(0, root.crosshairPos.y - 10, 60, 20)
                    ctx.strokeRect(0, root.crosshairPos.y - 10, 60, 20)
                    
                    // 日期标签背景
                    ctx.fillRect(root.crosshairPos.x - 30, height - 20, 60, 20)
                    ctx.strokeRect(root.crosshairPos.x - 30, height - 20, 60, 20)
                    
                    // 价格文本
                    ctx.fillStyle = root.textColor
                    ctx.font = "10px sans-serif"
                    ctx.textAlign = "left"
                    ctx.fillText(price.toFixed(2), 5, root.crosshairPos.y + 5)
                    
                    // 日期文本
                    ctx.textAlign = "center"
                    ctx.fillText(Qt.formatDate(date, "yy.MM.dd"), root.crosshairPos.x, height - 5)
                }
            }
        }
    }

    PriceScale {
        minPrice: canvas.minPrice
        maxPrice: canvas.maxPrice
        priceRange: canvas.priceRange
        textColor: root.textColor
    }

    DateScale {
        visibleData: root.visibleData
        textColor: root.textColor
    }
}