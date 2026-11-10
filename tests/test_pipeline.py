import numpy as np
import pytest
from lungct.dataset import label_from_path, similarity_groups
from lungct.models import CANDIDATES, choose_patches, feature_representation, assert_group_independence


def test_original_spelling_and_classes():
    assert label_from_path('dataset/Bengin cases/a.jpg')==0
    assert label_from_path('dataset/Malignant cases/a.jpg')==1
    assert label_from_path('dataset/Normal cases/a.jpg')==2
    assert label_from_path('foo/no_labels.jpg') is None


def test_similarity_chain_is_one_group():
    h=[0b0,0b1,0b11,0b111000]
    g=similarity_groups(h,cutoff=1)
    assert g[0]==g[1]==g[2]
    assert g[3]!=g[0]


def test_prevent_leakage():
    group=np.array([1,2,3,1])
    with pytest.raises(ValueError):assert_group_independence(group,np.array([0,1]),np.array([3]))
    assert_group_independence(group,np.array([0,3]),np.array([1,2]))


def test_candidate_list_and_selection():
    assert sum(map(len,CANDIDATES.values()))==17
    r=np.array([.3,.1,.2,.4]);d=np.zeros((4,4))
    assert choose_patches(r,d,'relevance',2).tolist()==[0,3]
    tr=np.arange(16,dtype=float).reshape(2,4,2)
    te=np.arange(8,dtype=float).reshape(1,4,2)
    a,b=feature_representation(tr,te,'mean',0)
    assert a.shape==(2,2) and b.shape==(1,2)
    a,b=feature_representation(tr,te,'all',4)
    assert a.shape==(2,8) and b.shape==(1,8)


def test_random_selection_deterministic():
    a=np.arange(256).reshape(1,64,4)
    x1,_=feature_representation(a,a,'random',20,seed=42,fold_tag=901)
    x2,_=feature_representation(a,a,'random',20,seed=42,fold_tag=901)
    np.testing.assert_array_equal(x1,x2)
