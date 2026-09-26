"""Executable catalog validator tests; source export remains a separate Rust gate."""
from pathlib import Path
import copy,json,tempfile,unittest
import gemma4_catalog as catalog
TOOLS=Path(__file__).resolve().parent
class CatalogTests(unittest.TestCase):
    def setUp(self):self.value=json.loads((TOOLS/'gemma4_metal_catalog.json').read_text())
    def test_complete_catalog_and_entrypoint_map(self):
        got=catalog.validate(self.value)
        self.assertEqual(len(got['candidates']),48)
        self.assertEqual(len(catalog.exports(got)),48)
        self.assertEqual(len(catalog.sources(got)),55)
        self.assertEqual(catalog.exports(got)['metal-mma32-load4'],('wave2_gemm_mma32_load4','wave2_qkv_mma32_load4'))
        self.assertEqual(len(catalog.exports(got)['metal-donor12b-sg8']),12)
    def test_duplicate_names_missing_families_and_unknown_fields_are_rejected(self):
        for change in [lambda v:v['candidates'].pop(),lambda v:v['candidates'].append(v['candidates'][1]),
                       lambda v:v.update(device_qualified=True),lambda v:v.update(default='auto'),
                       lambda v:v['candidates'][1].update(min_tokens=True),lambda v:v.update(extra=1)]:
            v=copy.deepcopy(self.value);change(v)
            with self.assertRaises(ValueError):catalog.validate(v)
    def test_bad_paths_limits_and_foreign_entrypoints_are_rejected(self):
        for value in ['../escape.metal','crates/rvllm-apple-metal/src/research_shaders/../x.metal','/tmp/x.metal','a;echo.metal']:
            v=copy.deepcopy(self.value);v['candidates'][1]['source_file']=value
            with self.assertRaises(ValueError):catalog.validate(v)
        for key,value in [('threads',33),('source_shared_bytes',32769),('threads',True)]:
            v=copy.deepcopy(self.value);v['candidates'][1]['budgets'][0][key]=value
            with self.assertRaises(ValueError):catalog.validate(v)
        v=copy.deepcopy(self.value);v['candidates'][1]['kernels'][0]='arbitrary'
        with self.assertRaises(ValueError):catalog.validate(v)
    def test_duplicate_json_key_and_nonfinite_values_are_errors(self):
        for text in ['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}']:
            with self.assertRaises(ValueError):catalog.decode(text)
    def test_exported_catalog_must_match_entire_reviewed_contract(self):
        catalog.verify_exported(self.value,self.value)
        v=copy.deepcopy(self.value);v['candidates'][1]['max_tokens']+=1
        with self.assertRaises(ValueError):catalog.verify_exported(self.value,v)
    def test_sources_reject_missing_duplicate_and_foreign_kernel_symbols(self):
        kind='metal-mma32-f32';names=catalog.exports(self.value)[kind]
        src='// unrelated baseline declaration\nkernel void ordinary() {}\n'+'\n'.join(f'kernel void {x}() {{}}' for x in names)
        catalog.check_source(self.value,kind,src)
        for mutant in ['',src.replace(names[0],'not_a_research_kernel'),src+f'\nkernel void {names[0]}() {{}}',src+'\nkernel void research_other() {}']:
            with self.assertRaises(ValueError):catalog.check_source(self.value,kind,mutant)
    def test_regular_catalog_bytes_are_bounded_and_leaf_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'catalog.json';p.write_text(json.dumps(self.value))
            self.assertEqual(catalog.load(p),self.value)
            alias=Path(d)/'alias.json';alias.symlink_to(p)
            with self.assertRaises(ValueError):catalog.load(alias)
            p.write_bytes(b' '* (catalog.MAX_JSON+1))
            with self.assertRaises(ValueError):catalog.load(p)
if __name__=='__main__':unittest.main()
