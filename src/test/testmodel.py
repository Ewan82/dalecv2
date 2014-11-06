from model import model as m
from model import data as dC



def test_acm():
    d=dC.dalecData(201)
    assert m.acm(0.01, d.p17, d.p11, d, 200) < 1e-3