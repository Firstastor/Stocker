import QtQuick
import QtQuick.Controls
import QtQuick.Controls.FluentWinUI3 
import QtQuick.Layouts
import QtQuick.Window  

Page {
    id: root
    property string sortField: ""
    property bool sortAscending: true
    signal stockSelected(string code, string name)
    // 根据屏幕尺寸调整的列宽比例
    readonly property real codeWidthRatio: Screen.width > 1920 ? 0.15 : 0.2
    readonly property real nameWidthRatio: Screen.width > 1920 ? 0.25 : 0.2
    readonly property real priceWidthRatio: 0.2
    readonly property real changePercentWidthRatio: 0.2
    readonly property real changeAmountWidthRatio: 0.2

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // 搜索框
        TextField {
            id: searchField
            Layout.fillWidth: true
            placeholderText: qsTr("搜索股票代码或名称...")
            selectByMouse: true
            leftPadding: 10
            rightPadding: 10
            topPadding: 8
            bottomPadding: 8
            onTextChanged: {
                StockInfoData.setFilterString(text)
            }
            background: Rectangle {
                color: "transparent"
                border.color: palette.mid
                radius: 4
            }
        }

        // 表头
        RowLayout {
            id: headerRow
            Layout.fillWidth: true
            spacing: 1
            height: 40

            // 代码表头
            HeaderButton {
                text: qsTr("代码")
                Layout.preferredWidth: root.width * codeWidthRatio
                sortField: "代码"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("代码")
            }

            // 名称表头
            HeaderButton {
                text: qsTr("名称")
                Layout.preferredWidth: root.width * nameWidthRatio
                sortField: "名称"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("名称")
            }

            // 最新价表头
            HeaderButton {
                text: qsTr("最新价")
                Layout.preferredWidth: root.width * priceWidthRatio
                sortField: "最新价"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("最新价")
            }

            // 涨幅表头
            HeaderButton {
                text: qsTr("涨幅")
                Layout.preferredWidth: root.width * changePercentWidthRatio
                sortField: "涨幅"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("涨幅")
            }

            // 涨跌表头
            HeaderButton {
                text: qsTr("涨跌")
                Layout.preferredWidth: root.width * changeAmountWidthRatio
                sortField: "涨跌"
                currentSortField: root.sortField
                ascending: root.sortAscending
                onClicked: updateSort("涨跌")
            }
        }

        // 股票列表
        ListView {
            id: listView
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 1
            clip: true
            model: StockInfoData
            
            // 性能优化设置
            cacheBuffer: listView.height * 2  // 缓存两屏高度的项目
            boundsBehavior: Flickable.StopAtBounds
            maximumFlickVelocity: 1500
            flickDeceleration: 2000
        

            delegate: Item {
                id: delegateItem
                width: ListView.view.width
                height: 40
                
                // 使用Loader实现懒加载
                Loader {
                    id: itemLoader
                    anchors.fill: parent
                    sourceComponent: actualDelegate
                    active: listView.visible
                }
                
                Component {
                    id: actualDelegate
                    Rectangle {
                        id: contentItem
                        anchors.fill: parent
                        color: mouseArea.containsMouse ? Qt.lighter(palette.highlight, 1.5) : palette.base
                        border.width: 1
                        border.color: palette.mid
                        
                        // 简化的鼠标效果
                        scale: mouseArea.pressed ? 0.98 : 1.0
                        Behavior on scale { NumberAnimation { duration: 80 } }
                        Behavior on color { ColorAnimation { duration: 150 } }

                        MouseArea {
                            id: mouseArea
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: root.stockSelected(model.代码, model.名称)
                        }

                        RowLayout {
                            anchors.fill: parent
                            spacing: 1

                            // 代码列
                            Text {
                                Layout.preferredWidth: root.width * codeWidthRatio
                                text: model.代码
                                color: mouseArea.containsMouse ? palette.highlightedText : palette.text
                                elide: Text.ElideRight
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                                font.pixelSize: 12
                            }

                            // 名称列
                            Text {
                                Layout.preferredWidth: root.width * nameWidthRatio
                                text: model.名称
                                color: mouseArea.containsMouse ? palette.highlightedText : palette.text
                                elide: Text.ElideRight
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                                font.pixelSize: 12
                            }

                            // 最新价列
                            Text {
                                Layout.preferredWidth: root.width * priceWidthRatio
                                text: model.最新价
                                elide: Text.ElideRight
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                                font.pixelSize: 12
                                font.bold: true
                                color: getPriceColor(model.涨跌, mouseArea.containsMouse)
                            }

                            // 涨幅列
                            Text {
                                Layout.preferredWidth: root.width * changePercentWidthRatio
                                text: model.涨幅 + "%"
                                elide: Text.ElideRight
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                                font.pixelSize: 12
                                font.bold: true
                                color: getPriceColor(model.涨幅, mouseArea.containsMouse)
                            }

                            // 涨跌列
                            Text {
                                Layout.preferredWidth: root.width * changeAmountWidthRatio
                                text: model.涨跌
                                elide: Text.ElideRight
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                                font.pixelSize: 12
                                font.bold: true
                                color: getPriceColor(model.涨跌, mouseArea.containsMouse)
                            }
                        }
                    }
                }
            }

            // 空状态提示
            Label {
                anchors.centerIn: parent
                visible: listView.count === 0
                text: qsTr("没有找到匹配的股票")
                font.pixelSize: 16
                color: palette.mid
            }
        }
    }

    component HeaderButton : Button {
        property string sortField: ""
        property string currentSortField: ""
        property bool ascending: true

        flat: true
        font.bold: true
        
        // 将文本生成逻辑提取到单独的属性中
        readonly property string displayText: {
            if (sortField === currentSortField) {
                return text + (ascending ? " ↑" : " ↓")
            }
            return text
        }

        contentItem: Text {
            text: parent.displayText
            font: parent.font
            color: parent.down ? palette.highlight : palette.text
            horizontalAlignment: parent.horizontalAlignment
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
            textFormat: Text.RichText
        }

        background: Rectangle {
            color: parent.down ? Qt.darker(palette.mid, 1.2) : 
                parent.hovered ? Qt.lighter(palette.mid, 1.5) : palette.mid
            radius: 2
        }
    }

    // 更新排序逻辑
    function updateSort(field) {
        if (root.sortField === field) {
            root.sortAscending = !root.sortAscending
        } else {
            root.sortField = field
            root.sortAscending = true
        }
        StockInfoData.sortByField(root.sortField, root.sortAscending)
    }

    // 获取价格颜色
    function getPriceColor(value, isHovered) {
        if (value > 0) return "red"
        if (value < 0) return "green" 
        return isHovered ? palette.highlightedText : palette.text
    }
}