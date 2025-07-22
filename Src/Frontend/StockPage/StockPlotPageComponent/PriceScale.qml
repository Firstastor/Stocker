import QtQuick
import QtQuick.Layouts

Column {
    id: root
    property real minPrice: 0
    property real maxPrice: 0
    property real priceRange: 0
    property color textColor: palette.text
    
    anchors {
        left: parent.left
        top: parent.top
        bottom: parent.bottom
        margins: 5
    }
    width: 60
    spacing: 0

    Repeater {
        model: 6
        Text {
            width: parent.width
            height: parent.height / 5
            text: (root.minPrice + (5-index) * root.priceRange/5).toFixed(2)
            color: root.textColor
            font.pixelSize: 10
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: index === 0 ? Text.AlignTop : 
                              index === 5 ? Text.AlignBottom : Text.AlignVCenter
        }
    }
}