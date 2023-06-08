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
    
class Lego(Scene):
    def construct(self):
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        tex = Tex(r'$\begin{bmatrix}\
                  0 & 0 & 0 & 1 & 0 & 0 & 1\\\
                  0 & 0 & 0 & 1 & 0 & 0 & 1\\\
                  0 & 0 & 0 & 1\end{bmatrix}$',
                  tex_template=myTemplate,
                  font_size=144)
        self.add(tex)
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
    

class IRMgen(Scene):
    def construct(self):
        pass