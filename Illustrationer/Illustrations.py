# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:58:48 2023

@author: nikol
"""

from manim import *
import numpy as np
config.background_color = WHITE
'''
Kommandoer (til egen maskine...):
cd C:\\Users\\nikol\\Desktop\\Studie\\Projekt\\Animation
manim -pql AnimTest.py MovingDiGraph()
'''

class FlowNet(Scene):
    def construct(self):
        # Sinks
        def sink(level):
            bucket=Polygon([-0.55,0,0],[-0.55,1,0],[-0.45,1,0],[-0.45,0.1,0],
                           [0.45,0.1,0],[0.45,1,0],[0.55,1,0],[0.55,0,0],
                           color=GREY, sheen_factor=0)
            bucket.set_fill(GREY, opacity=1)
            
            water=Polygon([-0.4,0.15,0],[-0.4,0.15+level,0],
                           [0.4,0.15+level,0],[0.4,0.15,0],color=BLUE_E,
                           sheen_factor=0)
            water.set_fill(BLUE_E, opacity=1)
            
            
            return VGroup(bucket,water).scale(0.7)
        
        # Flow Network
        vertices = [i for i in range(1,11)]
        edges = [(1,2), (1,3), (2,4), (2,5), (3,5), (3,6),
                 (4,7), (5,8), (6,8), (6,9), (6,10)]
        
        lt = {1: [0,2,0], 2: [-1,0,0], 3: [1,0,0], 
              4: [-2,-2,0], 5: [0,-2,0], 6: [2,-2,0], 
              7: [-2.5,-4,0], 8: [-0.5,-4,0], 9: [1,-4,0], 
              10: [3,-4,0]}
        
        vertex_config = {'radius': 0.2, "fill_color": GREY}
        
        edge_config = {(1,2): {"stroke_width": 12, "stroke_color": BLUE_E},
                       (1,3): {"stroke_width": 9, "stroke_color": BLUE_E},
                       (2,4): {"stroke_width": 5, "stroke_color": BLUE_E},
                       (2,5): {"stroke_width": 9, "stroke_color": BLUE_E},
                       (3,5): {"stroke_width": 3, "stroke_color": BLUE_E},
                       (3,6): {"stroke_width": 7, "stroke_color": BLUE_E},
                       (4,7): {"stroke_width": 5, "stroke_color": BLUE_E},
                       (5,8): {"stroke_width": 10, "stroke_color": BLUE_E},
                       (6,8): {"stroke_width": 3, "stroke_color": BLUE_E},
                       (6,9): {"stroke_width": 3, "stroke_color": BLUE_E},
                       (6,10): {"stroke_width": 3, "stroke_color": BLUE_E},
                       "tip_config": {"tip_shape": StealthTip,
                                      "tip_length": 0.15}}
        
        sink7 = sink(0.2).move_to([-2.4,-3,0])
        sink8 = sink(0.6).move_to([-0.5,-3,0])
        sink9 = sink(0.1).move_to([1,-3,0])
        sink10 = sink(0.1).move_to([2.9,-3,0])
        
        vertex = {i: Dot(radius=0.3 , color = WHITE, fill_opacity=0.5) for i in range(7,11)}
        
        g = DiGraph(vertices,
                    edges,
                    layout=lt,
                    vertex_config=vertex_config,
                    vertex_mobjects=vertex,
                    edge_config=edge_config).shift(UP*1.5)
        
        self.add(sink7)
        self.add(sink8)
        self.add(sink9)
        self.add(sink10)
        self.add(g)
        
        # Text
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        tex = Tex('$e^{\pi i}+1=0$',
                  tex_template=myTemplate,
                  font_size=144)
        
        flows = {(1,2): (0.6, [-1,2,0]), (1,3): (0.4, [1,2,0]), 
                 (2,4): (0.2, [-1.8,0,0]), (2,5): (0.4, [-0.1,0,0]), 
                 (3,5): (0.1, [0.9,0,0]), (3,6): (0.3, [1.8,0,0]),
                 (4,7): (0.2, [-2.7,-2,0]), (5,8): (0.5, [-0.7,-2,0]), 
                 (6,9): (0.1, [1.8,-2.3,0]), (6,10): (0.1, [2.8,-2,0]),
                 (6,8): (0.1, [0.4,-2,0])}
        
        texts = []
        for edge in edges:
            text, pos = flows[edge]
            texts.append(MathTex(str(text),color=BLUE_E,font_size=34).move_to(pos).shift(UP*0.7))
        
        self.add(*texts)
        self.wait()
    
class Gibbs(Scene):
    def construct(self):
        vertices = [i for i in range(1,8)]
        edges = [(1,2),(1,3),(2,4),(3,4),(1,5),(3,6),
                 (5,6),(5,7),(6,7),(2,3)]
        
        lt = {1: [-1,2,0], 2: [-2,0,0], 3: [0,0,0],
              4: [-1,-2,0], 5: [2,2,0], 6: [2,0,0], 
              7: [3,1,0]}
        
        def ClusteredGraph(clustering, vertices, edges, layout):
            # Max of 6 colors
            edge_config = {"stroke_color": GREY}
            vertex_config = {"fill_color": GREY}
            cs = [RED, GREEN, BLUE, YELLOW_D, ORANGE, PURPLE]
            for i, cluster in enumerate(clustering):
                vertex_config.update({node: {"fill_color": cs[i]} for node in cluster})
                
                edge_config.update({edge: {"stroke_color": cs[i]} for edge in edges
                                    if set(edge).issubset(cluster)})
            
            g = Graph(vertices,
                        edges,
                        labels=True,
                        vertex_config=vertex_config,
                        edge_config=edge_config,
                        layout=lt).scale(scale)
            
            return g
        
        # Sizing
        scale=0.5
        fontsize=30
        
        clustering = [list(range(1,8))]
        
        g1 = ClusteredGraph(clustering, vertices, edges, lt)
        
        clustering = [[1,2,3,5,6,7],[4]]
        
        g2 = ClusteredGraph(clustering, vertices, edges, lt)
        
        clustering = [[1,2,3,5,6],[4],[7]]
        
        g3 = ClusteredGraph(clustering, vertices, edges, lt)
        
        clustering = [[1,2,5,6],[3,4],[7]]
        
        g4 = ClusteredGraph(clustering, vertices, edges, lt)
        
        clustering = [[1,2,3,4],[5,6,7]]
        
        g5 = ClusteredGraph(clustering, vertices, edges, lt)
        
        tex1 = MathTex(r"\xrightarrow{4?}", color=BLACK, font_size=fontsize)
        tex2 = MathTex(r"\xrightarrow{7?}", color=BLACK, font_size=fontsize)
        tex3 = MathTex(r"\xrightarrow{3?}", color=BLACK, font_size=fontsize)
        tex4 = MathTex(r"\xrightarrow{} \dots \xrightarrow{}", color=BLACK, font_size=fontsize)

        
        self.add(VGroup(g1,tex1,g2,tex2,g3).arrange(RIGHT, buff=0.2).move_to([0,2,0]))

        self.add(VGroup(tex3,g4,tex4,g5).arrange(RIGHT,buff=0.2).move_to([0,-2,0]))
        
        self.wait()
        pass
    
class ClusterMatrix(Scene):
    def construct(self):
        v = [1,2,3,4]
        e = [(1,2),(2,3),(2,4)]
        
        lt = {1: [-1,2,0], 2: [-2,0,0], 3: [0,0,0], 4: [-1,-2,0]}
        
        vertex_config = {"fill_color":GREY_C}
        edge_config={"stroke_color":GREY_C,"stroke_width":12}
        
        g = Graph(v,
                e,
                layout=lt,
                labels=True,
                vertex_config=vertex_config,
                edge_config=edge_config).scale(1).move_to([2,0,0])
        
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        adj_tex = Tex(r'$A=\begin{bmatrix}\
                      0 & 1 & 0 & 0\\\
                      1 & 0 & 1 & 1\\\
                      0 & 1 & 0 & 0\\\
                      0 & 1 & 0 & 0\end{bmatrix}$',
                      tex_template=myTemplate,
                      color=GREY_D,
                      font_size=40).move_to([-2.5,1.3,0])
            
        clu_tex = Tex(r'$C=\begin{bmatrix}\
                      1 & 1 & 1 & 0\\\
                      1 & 1 & 1 & 0\\\
                      1 & 1 & 1 & 0\\\
                      0 & 0 & 0 & 1\end{bmatrix}$',
                      tex_template=myTemplate,
                      color=GREY_D,
                      font_size=40).move_to([-2.5,-1.3,0])
            
        self.add(adj_tex, clu_tex, g)
        self.wait()

class CLusterFLow(Scene):
    def construct(self):
        v = [1,2,3]
        e = [(1,2),(1,3),(2,3)]
        
        lt = {1: [-1,2,0], 2: [-2,0,0], 3: [0,0,0]}
        
        def pclustered(v, e, lt, considered):
            vertex_config = {"fill_color":GREY_B}
            edge_config={"stroke_color":GREY_B,"stroke_width":16}
            for vertice in considered[0]:
                vertex_config[vertice]={"fill_color":GREY_C}
            
            for edge in considered[1]:
                edge_config[edge]={"stroke_color":GREY_C,"stroke_width":20}
                
            
            g = Graph(v,
                    e,
                    layout=lt,
                    labels=True,
                    vertex_config=vertex_config,
                    edge_config=edge_config).scale(2)
                
            
            return g
        
        g1 = pclustered(v,e,lt,[[1],[]])
        
        g2 = pclustered(v,e,lt,[[1,2],[(1,2)]])
        
        g3 = pclustered(v,e,lt,[v,e])
        
        self.add(g3)
        self.wait()
    

class IRMgen1(Scene):
    def construct(self):
        vertices = [i for i in range(1,10)]
        edges = []
        layout = {1: [-1.5,2,0], 2: [-0,2,0], 3: [-0.75,0.5,0], 
                  4: [-2.5,-2,0], 5: [-1.5,-3,0], 
                  6: [2.5,0,0], 7: [4,0,0], 8: [2.5,-1.5,0], 9: [4,-1.5,0]}
        
        c1_config = {i: {"fill_color": RED} for i in range(1,4)}
        c2_config = {i: {"fill_color": YELLOW_D} for i in range(4,6)}
        c3_config = {i: {"fill_color": BLUE} for i in range(6,10)}
        
        vertex_config = {**c1_config, **c2_config, **c3_config}
        
        g = Graph(vertices,
                  edges,
                  layout=layout,
                  labels=True,
                  vertex_config = vertex_config)
        
        self.add(g.scale(0.9).move_to(-0.4))
        
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        crp = MathTex(r"\text{CRP}(A) \longrightarrow z=[3,2,4]", color=BLACK, font_size=50)
        crp[0][11].set_color(RED)
        crp[0][13].set_color(YELLOW_D)
        crp[0][15].set_color(BLUE)
        self.add(crp.move_to([0,3,0]))
        self.wait()
        
        pass

class IRMgen2(Scene):
    def construct(self):
        vertices = [i for i in range(1,10)]
        
        edges_c1 = [(1,2),(1,3),(2,3)]
        edges_c2 = [(4,5)]
        edges_c3 = [(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]
        
        edges_c1c2 = [(i,j) for j in range(4,6) for i in range(1,4)]
        edges_c1c3 = [(i,j) for j in range(6,10) for i in range(1,4)]
        edges_c2c3 = [(i,j) for j in range(4,6) for i in range(6,10)]
        
        edgetypes = [edges_c1, edges_c2, edges_c3, 
                     edges_c1c2, edges_c1c3, edges_c2c3]
        
        all_edges = [edge for edges in edgetypes for edge in edges]
        
        e_colors = [RED, YELLOW_D, BLUE,
                    ORANGE, PURPLE, GREEN]
        
        edge_config = {edge: {"stroke_color": e_colors[i], "stroke_opacity":0.5}
                       for i, edges in enumerate(edgetypes) for edge in edges}
        
        
        layout = {1: [-1.5,2,0], 2: [-0,2,0], 3: [-0.75,0.5,0], 
                  4: [-2.5,-2,0], 5: [-1.5,-3,0], 
                  6: [2.5,0,0], 7: [4,0,0], 8: [2.5,-1.5,0], 9: [4,-1.5,0]}
        
        vertex_config = {"fill_color":GREY_B}
        
        g = Graph(vertices,
                  all_edges,
                  layout=layout,
                  labels=True,
                  vertex_config = vertex_config,
                  edge_config = edge_config)
        
        self.add(g.scale(0.9).move_to(-0.4))
        
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        beta = MathTex(r"\text{Beta}(a,b) \longrightarrow p(\phi_{k,l})=\Phi=\
                       \begin{bmatrix}\
                      0.9 & 0.1 & 0.7 \\\
                      0.1 & 0.8 & 0.5 \\\
                      0.7 & 0.5 & 0.2 \end{bmatrix}",
                      color=BLACK, font_size=50)
            
        self.add(beta.move_to([0,3,0]))
        
        text_colors = [RED, ORANGE, PURPLE,
                       ORANGE, YELLOW_D, GREEN,
                       PURPLE, GREEN, BLUE]
        
        for c, i in enumerate(range(23,48,3)):
            beta[0][i:i+3].set_color(text_colors[c])

        self.wait()
        
        pass

class IRMgen3(Scene):
    def construct(self):
        vertices = [i for i in range(1,10)]
        
        edges = [(1,2),(1,3),(2,3),(1,4),(3,4),
                 (4,5),(6,8),(6,9),
                 (2,6),(2,7),(2,8), (3,9), (1,8), (1,7),
                 (4,6),(4,7),(4,8),(5,7),(5,9)]
        
        edge_config = {"stroke_color": GREY_B}
        
        
        layout = {1: [-1.5,2,0], 2: [-0,2,0], 3: [-0.75,0.5,0], 
                  4: [-2.5,-2,0], 5: [-1.5,-3,0], 
                  6: [2.5,0,0], 7: [4,0,0], 8: [2.5,-1.5,0], 9: [4,-1.5,0]}
        
        vertex_config = {"fill_color":GREY_B}
        
        g = Graph(vertices,
                  edges,
                  layout=layout,
                  labels=True,
                  vertex_config = vertex_config,
                  edge_config = edge_config)
        
        self.add(g.scale(0.9).move_to(-0.4))
        
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        bern = MathTex(r"\text{Bernoulli}(\Phi) \longrightarrow X =",
                      color=BLACK, font_size=50)
        
        am =   MathTex(r"\begin{bmatrix}\
                      0 & 1 & \dots & 0 \\\
                      1 & 0 &       & 1 \\\
                      \vdots & & \ddots\\\
                      0 & 1 & \dots & 0  \end{bmatrix}",
                      color=BLACK, font_size=30)
            
        
        self.add(bern.move_to([-2,3,0]), am.move_to([2,3,0]))

        self.wait()
        
        pass
    
class IRMgenT(Scene):
    def construct(self):
        
        vertices = [i for i in range(1,10)]
        edges = []
        layout = {1: [-1.5,2,0], 2: [-0,2,0], 3: [-0.75,0.5,0], 
                  4: [-2.5,-2,0], 5: [-1.5,-3,0], 
                  6: [2.5,0,0], 7: [4,0,0], 8: [2.5,-1.5,0], 9: [4,-1.5,0]}
        
        c1_config = {i: {"fill_color": RED} for i in range(1,4)}
        c2_config = {i: {"fill_color": YELLOW_D} for i in range(4,6)}
        c3_config = {i: {"fill_color": BLUE} for i in range(6,10)}
        
        vertex_config = {**c1_config, **c2_config, **c3_config}
        
        g = Graph(vertices,
                  edges,
                  layout=layout,
                  labels=True,
                  vertex_config = vertex_config)
        
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        crp = MathTex(r"\text{CRP}(A) \longrightarrow z=[3,2,4]", color=BLACK, font_size=50)
        crp[0][11].set_color(RED)
        crp[0][13].set_color(YELLOW_D)
        crp[0][15].set_color(BLUE)
        
        irm1 = VGroup(g.scale(0.9).move_to([0,-1,0]), crp.move_to([0,3.5,0]))
        
        #####################################################
        
        vertices = [i for i in range(1,10)]
        
        edges_c1 = [(1,2),(1,3),(2,3)]
        edges_c2 = [(4,5)]
        edges_c3 = [(6,7),(6,8),(6,9),(7,8),(7,9),(8,9)]
        
        edges_c1c2 = [(i,j) for j in range(4,6) for i in range(1,4)]
        edges_c1c3 = [(i,j) for j in range(6,10) for i in range(1,4)]
        edges_c2c3 = [(i,j) for j in range(4,6) for i in range(6,10)]
        
        edgetypes = [edges_c1, edges_c2, edges_c3, 
                     edges_c1c2, edges_c1c3, edges_c2c3]
        
        all_edges = [edge for edges in edgetypes for edge in edges]
        
        e_colors = [RED, YELLOW_D, BLUE,
                    ORANGE, PURPLE, GREEN]
        
        edge_config = {edge: {"stroke_color": e_colors[i], "stroke_opacity":0.5}
                       for i, edges in enumerate(edgetypes) for edge in edges}
        
        
        layout = {1: [-1.5,2,0], 2: [-0,2,0], 3: [-0.75,0.5,0], 
                  4: [-2.5,-2,0], 5: [-1.5,-3,0], 
                  6: [2.5,0,0], 7: [4,0,0], 8: [2.5,-1.5,0], 9: [4,-1.5,0]}
        
        vertex_config = {"fill_color":GREY_B}
        
        g = Graph(vertices,
                  all_edges,
                  layout=layout,
                  labels=True,
                  vertex_config = vertex_config,
                  edge_config = edge_config)
        
        
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        beta = MathTex(r"\text{Beta}(a,b) \longrightarrow p(\phi_{k,l})=\Phi=\
                       \begin{bmatrix}\
                      0.9 & 0.1 & 0.7 \\\
                      0.1 & 0.8 & 0.5 \\\
                      0.7 & 0.5 & 0.2 \end{bmatrix}",
                      color=BLACK, font_size=50)
            
        self.add(beta.move_to([0,3,0]))
        
        text_colors = [RED, ORANGE, PURPLE,
                       ORANGE, YELLOW_D, GREEN,
                       PURPLE, GREEN, BLUE]
        
        for c, i in enumerate(range(23,48,3)):
            beta[0][i:i+3].set_color(text_colors[c])
        
        irm2 = VGroup(beta, g.scale(0.9).move_to(-0.4))
        
        #########################################################
        
        vertices = [i for i in range(1,10)]
        
        edges = [(1,2),(1,3),(2,3),(1,4),(3,4),
                 (4,5),(6,8),(6,9),
                 (2,6),(2,7),(2,8), (3,9), (1,8), (1,7),
                 (4,6),(4,7),(4,8),(5,7),(5,9)]
        
        edge_config = {"stroke_color": GREY_B}
        
        
        layout = {1: [-1.5,2,0], 2: [-0,2,0], 3: [-0.75,0.5,0], 
                  4: [-2.5,-2,0], 5: [-1.5,-3,0], 
                  6: [2.5,0,0], 7: [4,0,0], 8: [2.5,-1.5,0], 9: [4,-1.5,0]}
        
        vertex_config = {"fill_color":GREY_B}
        
        g = Graph(vertices,
                  edges,
                  layout=layout,
                  labels=True,
                  vertex_config = vertex_config,
                  edge_config = edge_config)
        
        
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        bern = MathTex(r"\text{Bernoulli}(\Phi) \longrightarrow X =",
                      color=BLACK, font_size=50)
        
        am =   MathTex(r"\begin{bmatrix}\
                      0 & 1 & \dots & 0 \\\
                      1 & 0 &       & 1 \\\
                      \vdots & & \ddots\\\
                      0 & 1 & \dots & 0  \end{bmatrix}",
                      color=BLACK, font_size=30)
            
        irm3 = VGroup(g.scale(0.9).move_to(-0.4), 
                      bern.move_to([-2,3,0]), 
                      am.move_to([2,3,0]))
        
        ######################################################
        
        scale = 0.3
        
        self.add(VGroup(irm1.scale(scale),irm2.scale(scale),irm3.scale(scale)).arrange(RIGHT))
        self.wait()
        
        