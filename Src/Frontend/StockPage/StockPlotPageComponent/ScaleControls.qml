import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3
import QtQuick.Layouts

RowLayout {
    id: root
    property int currentScale: 240
    property color bgColor: palette.base
    property color textColor: palette.text
    property color gridColor: "#e0e0e0"
    height: 40
    spacing: 1

    signal scaleSelected(int newScale)

    ButtonGroup { id: scaleGroup }

    Repeater {
        model: ["日K", "5", "15", "30", "60"]
        Button {
            text: modelData
            flat: true
            checked: [240, 5, 15, 30, 60][index] === root.currentScale
            ButtonGroup.group: scaleGroup
            onClicked: {
                var newScale = [240, 5, 15, 30, 60][index]
                root.scaleSelected(newScale)
                root.currentScale = newScale
            }
        }
    }
}