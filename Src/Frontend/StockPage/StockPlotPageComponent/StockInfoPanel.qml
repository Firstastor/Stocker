import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3
import QtQuick.Layouts

GridLayout {
    id: root
    property color textColor: palette.text
    columns: 6
    rows: 3
    columnSpacing: 10

    // 第一列
    Label { 
        text: "最高" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        id: highPrice
        text: "1446.80" 
        color: root.textColor 
    }
    Label { 
        text: "总股本" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        text: "1.81T" 
        color: root.textColor 
    }
    Label { 
        text: "成交量" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        text: "262.0310k" 
        color: root.textColor 
    }

    // 第二列
    Label { 
        text: "最低" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        id: lowPrice
        text: "1435.00" 
        color: root.textColor 
    }
    Label { 
        text: "流通股本" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        text: "1.81T" 
        color: root.textColor 
    }
    Label { 
        text: "成交额" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        text: "37.80100M" 
        color: root.textColor 
    }

    // 第三列
    Label { 
        text: "今开" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        id: openPrice
        text: "1436.99" 
        color: root.textColor 
    }
    Label { 
        text: "市盈率" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        text: "16.88%" 
        color: root.textColor 
    }
    Label { 
        text: "换手率" 
        font.bold: true 
        color: root.textColor 
    }
    Label { 
        text: "0.21%" 
        color: root.textColor 
    }

    function updateInfoData() {
        for (var i = 0; i < StockInfoData.rowCount(); i++) {
            var item = StockInfoData.get(i)
            if (item.代码 === stockCode) {
                highPrice.text = item.最高
                lowPrice.text = item.最低
                openPrice.text = item.今开
                break
            }
        }
    }

    Component.onCompleted: updateInfoData()
}