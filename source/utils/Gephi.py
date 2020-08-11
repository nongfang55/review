# coding=utf-8

import jpype
import os
from jpype import JClass
import xml.dom.minidom

from source.config.configPraser import configPraser
from source.config.projectConfig import projectConfig


class Gephi:
    Lookup = None

    def __init__(self):
        """使用jpype开启虚拟机（在开启jvm之前要加载类路径）"""

        # 依赖jar包路径
        gephi_jar_path = projectConfig.getRootPath() + os.sep + 'source' \
                         + os.sep + 'utils' + os.sep + 'gephi-toolkit-0.9.3.jar'
        # java虚拟机的路径
        jvm_path = configPraser.getJVMPath()

        try:
            # 开启jvm
            jpype.startJVM(jvm_path, "-ea", "-Djava.class.path=%s" % (gephi_jar_path))
        except:
            pass

        # 初始化lookup
        self.Lookup = self.getClass("org.openide.util.Lookup")


    def getObject(self, class_path):
        Class = JClass(class_path)
        return Class()

    def getClass(self, class_path):
        return JClass(class_path)

    def lookup(self, name, namespace='org.gephi.'):
        Class = jpype.java.lang.Class.forName(namespace + name)
        return self.Lookup.getDefault().lookup(Class)

    def getCommunity(self, graph_file="", out_file="tmp.gexf", layout=False, save_pdf=False, save_gephi=False):
        print("loading graph......")
        """引入必要类"""
        projectController = self.lookup('project.api.ProjectController')
        importController = self.lookup("io.importer.api.ImportController")
        graphController = self.lookup('graph.api.GraphController')
        exportController = self.lookup('io.exporter.api.ExportController')

        """创建工作空间"""
        projectController.newProject()
        workspace = projectController.getCurrentWorkspace()

        """导入graph文件"""
        graph = importController.importFile(jpype.java.io.File(graph_file))
        edge_default = graph.getEdgeDefault()
        graph.setEdgeDefault(edge_default.values()[0])  # DIRECTED, UNDIRECTED, MIXED
        defaultProcessor = self.getClass('org.gephi.io.processor.plugin.DefaultProcessor')
        importController.process(graph, defaultProcessor(), workspace)

        """获取graph"""
        graphModel = graphController.getGraphModel()
        jpype.java.lang.System.out.println("Nodes: " + str(graph.getNodeCount()))
        jpype.java.lang.System.out.println("Edges: " + str(graph.getEdgeCount()))

        """社区发现"""
        modularityClass = self.getClass('org.gephi.statistics.plugin.Modularity')
        modularityObject = modularityClass()
        modularityObject.setRandom(True)
        modularityObject.setUseWeight(True)
        modularityObject.setResolution(1)
        modularityObject.execute(graphModel)
        modularity = modularityObject.getModularity()

        """导出graph"""
        csvExporter = exportController.getExporter("gexf")
        csvExporter.setWorkspace(workspace)
        csvExporter.setExportVisible(True)
        exportController.exportFile(jpype.java.io.File(out_file), csvExporter)

        """解析表，获得分类"""
        dom = xml.dom.minidom.parse(out_file)
        dom_root = dom.documentElement
        dom_nodes = dom_root.getElementsByTagName('nodes')[0]
        nodes = dom_nodes.childNodes
        community = {}
        for node in nodes:
            if node is None or node.nodeName != 'node':
                continue
            attrs = node.getElementsByTagName('attvalues')[0].childNodes
            for attr in attrs:
                if attr is None or attr.nodeName != 'attvalue':
                    continue
                """目前只关注modularity"""
                if attr.getAttribute('for') == 'modularity_class':
                    community_id = attr.getAttribute('value')
                    if not community.__contains__(community_id):
                        community[community_id] = []
                    community[community_id].append(node.getAttribute('id'))
                    break

        """删除中间结果"""
        os.remove(out_file)

        return community, modularity


if __name__ == '__main__':
    g = Gephi()
    projectConfig.getRootPath()
    g.layout(graph_file="D:/review/review/source/utils/opencv_2017_1_2018_9_network.gexf",
             out_file="tmp.gexf")
