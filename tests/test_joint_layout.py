from ml_architecture_diagram.joint_layout import optimize_publication_layout, score_layout
from ml_architecture_diagram.routing import fanout_bundle_geometry


def test_joint_optimizer_removes_simple_lane_crossing():
    spec={
        'nodes':[
            {'id':'a','group':'stage_inputs'}, {'id':'b','group':'stage_inputs'},
            {'id':'c','group':'stage_encoders'}, {'id':'d','group':'stage_encoders'},
        ],
        'groups':[{'id':'stage_inputs'},{'id':'stage_encoders'}],
    }
    pos={
        'a':{'x':0,'y':0,'w':80,'h':50}, 'b':{'x':0,'y':100,'w':80,'h':50},
        'c':{'x':220,'y':0,'w':80,'h':50}, 'd':{'x':220,'y':100,'w':80,'h':50},
    }
    edges=[{'from':'a','to':'d','type':'main'},{'from':'b','to':'c','type':'main'}]
    before=score_layout(spec,pos,edges)[1]
    _,_,_,meta=optimize_publication_layout(spec,pos,edges,{'framework_core_mode':'horizontal'})
    assert before['edge_crossings'] == 1
    assert meta['after']['edge_crossings'] == 0
    assert meta['after']['objective'] < meta['before']['objective']
    assert meta['improvement_percent'] > 0


def test_fanout_bundle_preserves_one_branch_per_target():
    boxes={
        's':{'x':0,'y':50,'w':80,'h':50},
        'a':{'x':220,'y':0,'w':80,'h':50},
        'b':{'x':220,'y':70,'w':80,'h':50},
        'c':{'x':220,'y':140,'w':80,'h':50},
    }
    edges=[{'from':'s','to':t,'type':'main'} for t in ('a','b','c')]
    geo=fanout_bundle_geometry(edges,boxes,'s')
    assert geo is not None
    assert len(geo['branches']) == 3
    assert {br['edge']['to'] for br in geo['branches']} == {'a','b','c'}
    assert len(geo['trunk']) >= 2
    assert len(geo['bus']) == 2
