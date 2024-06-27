import pytest

class TestInit():

  def test_import(self):
    successful = False
    try:
      import ngclearn
      successful = True
    except:
      successful = False
    assert successful, "Cannot import ngclearn"

