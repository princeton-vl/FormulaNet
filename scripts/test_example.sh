# Commands to test pretrained models.
cd src
python batch_test.py --model FormulNet-basic --log ../logs/test_formulanet_basic.log --compatible 
python batch_test.py --model FormulNet-basic-uc --log ../logs/test_formulanet_basic_uc.log --compatible 
python batch_test.py --model FormulNet --log ../logs/test_formulanet.log 
python batch_test.py --model FormulNet-uc --log ../logs/test_formulanet_uc.log
