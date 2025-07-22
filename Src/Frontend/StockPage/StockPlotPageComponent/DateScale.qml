import QtQuick
import QtQuick.Layouts

Row {
    id: root
    property var visibleData: []
    property color textColor: "black"
    
    anchors {
        left: parent.left
        right: parent.right
        bottom: parent.bottom
        margins: 5
    }
    height: 20
    spacing: 0

    Repeater {
        model: 7
        Text {
            width: parent.width / 6
            height: parent.height
            text: {
                if (root.visibleData.length === 0) return ""
                var idx = Math.floor(index * (root.visibleData.length-1) / 6)
                return Qt.formatDate(new Date(root.visibleData[idx].日期), "MM/dd")
            }
            color: root.textColor
            font.pixelSize: 10
            horizontalAlignment: Text.AlignHCenter
        }
    }
}