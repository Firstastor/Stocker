import QtQuick
import QtQuick.Controls 
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts
import QtQuick.Window 

Rectangle {
    color: palette.window
    signal switchPage(int index)
    property int currentIndex: 0  // 添加当前索引属性

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        Button {
            Layout.alignment: Qt.AlignCenter
            display: AbstractButton.TextUnderIcon
            text: qsTr("全部股票")
            flat: true
            highlighted: parent.parent.currentIndex === 0
            onClicked: {
                parent.parent.currentIndex = 0
                switchPage(0)
            }
        }

        Button {
            Layout.alignment: Qt.AlignCenter
            display: AbstractButton.TextUnderIcon
            text: qsTr("自选股")
            flat: true
            highlighted: parent.parent.currentIndex === 1
            onClicked: {
                parent.parent.currentIndex = 1
                switchPage(1)
            }
        }

        Button {
            Layout.alignment: Qt.AlignCenter
            display: AbstractButton.TextUnderIcon
            text: qsTr("回测")
            flat: true
            highlighted: parent.parent.currentIndex === 2
            onClicked: {
                parent.parent.currentIndex = 2
                switchPage(2)
            }
        }

        Item {
            Layout.fillHeight: true
            Layout.fillWidth: true
        }
        Button {
            Layout.alignment: Qt.AlignCenter
            display: AbstractButton.TextUnderIcon
            text: qsTr("设置")
            flat: true
            highlighted: parent.parent.currentIndex === 3
            onClicked: {
                parent.parent.currentIndex = 3
                switchPage(3)
            }
        }
    }
}