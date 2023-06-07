# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:58:48 2023

@author: nikol
"""

from manim import *

'''
Kommandoer (til egen maskine...):
cd C:\\Users\\nikol\\Desktop\\Studie\\Projekt\\Animation
manim -pql AnimTest.py MovingDiGraph()
'''

class FlowNet(Scene):
    def construct(self):
        # Sinks
        def sink(level,op):
            bucket=Polygon([-0.55,0,0],[-0.55,1,0],[-0.45,1,0],[-0.45,0.1,0],
                           [0.45,0.1,0],[0.45,1,0],[0.55,1,0],[0.55,0,0],
                           color=GREY, sheen_factor=0)
            bucket.set_fill(GREY, opacity=1)
            
            water=Polygon([-0.4,0.15,0],[-0.4,0.15+level,0],
                           [0.4,0.15+level,0],[0.4,0.15,0],color=BLUE,
                           sheen_factor=0)
            water.set_fill(BLUE, opacity=1)
            
            
            return VGroup(bucket,water).scale(0.7).set_opacity(op)
        
        # Flow Network
        vertices = [i for i in range(1,11)]
        edges = [(1,2), (1,3), (2,4), (2,5), (3,5), (3,6),
                 (4,7), (5,8), (6,9), (6,10)]
        
        lt = {1: [0,2,0], 2: [-1,0,0], 3: [1,0,0], 
              4: [-2,-2,0], 5: [0,-2,0], 6: [2,-2,0], 
              7: [-3,-4,0], 8: [-0.5,-4,0], 9: [1,-4,0], 
              10: [3,-4,0]}
        
        vertex_config = {'radius': 0.2}
        
        edge_config = {(1,2): {"stroke_width": 12, "stroke_color": BLUE},
                       (1,3): {"stroke_width": 8, "stroke_color": BLUE},
                       (2,4): {"stroke_width": 4, "stroke_color": BLUE},
                       (2,5): {"stroke_width": 8, "stroke_color": BLUE},
                       (3,5): {"stroke_width": 2, "stroke_color": BLUE},
                       (3,6): {"stroke_width": 6, "stroke_color": BLUE},
                       (4,7): {"stroke_width": 4, "stroke_color": BLUE},
                       (5,8): {"stroke_width": 10, "stroke_color": BLUE},
                       (6,9): {"stroke_width": 2, "stroke_color": BLUE},
                       (6,10): {"stroke_width": 4, "stroke_color": BLUE},
                       "tip_config": {"tip_shape": StealthTip,
                                      "tip_length": 0.15}}
        
        sink7 = sink(0.2, 1.0)
        sink8 = sink(0.5, 1.0)
        sink9 = sink(0.1, 1.0)
        sink10 = sink(0.2, 1.0)
        vertex = {7: sink7,8: sink8,9: sink9,10: sink10}
        
        g = DiGraph(vertices,
                    edges,
                    layout=lt,
                    vertex_config=vertex_config,
                    vertex_mobjects=vertex,
                    edge_config=edge_config).shift(UP*1.5)
        
        self.add(g)
        
        # Text
        myTemplate = TexTemplate()
        myTemplate.add_to_preamble(r"\usepackage{amsmath}")
        tex = Tex('$e^{\pi i}+1=0$',
                  tex_template=myTemplate,
                  font_size=144)
        
        flows = {(1,2): (0.6, [-1,2,0]), (1,3): (0.4, [1,2,0]), 
                 (2,4): (0.2, [-1.8,0,0]), (2,5): (0.4, [-0.3,0,0]), 
                 (3,5): (0.1, [0.3,0,0]), (3,6): (0.3, [1.8,0,0]),
                 (4,7): (0.2, [-2.7,-2,0]), (5,8): (0.5, [-0.7,-2,0]), 
                 (6,9): (0.1, [1,-2,0]), (6,10): (0.2, [3,-2,0])}
        
        texts = []
        for edge in edges:
            text, pos = flows[edge]
            texts.append(MathTex(str(text),color=BLUE,font_size=34).move_to(pos).shift(UP*0.7))
        
        self.add(*texts)
        self.wait()
    
class Lego(Scene):
    def construct(self):
        pass

class CLusterFLow(Scene):
    def construct(self):
        p
    
class ClusteringMatrix(Scene):
    def construct(self):
        pass

class IRMgen(Scene):
    def construct(self):
        pass