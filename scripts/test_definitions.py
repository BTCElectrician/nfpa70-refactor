import unittest
from data_processing.definitions_processor import DefinitionsProcessor
from data_processing.definitions_models import Definition

class TestDefinitionsProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DefinitionsProcessor()
        
    def test_extract_definition(self):
        test_text = """Power Supply. (Static Power Supply) A Class 2 power supply 
        connected between circuits. (393) (CMP-18)
        Informational Note: See standards for details."""
        
        definition = self.processor.extract_definition(test_text)
        
        self.assertIsNotNone(definition)
        self.assertEqual(definition.term, "Power Supply")
        self.assertEqual(definition.alternative_terms, ["Static Power Supply"])
        self.assertIn("(CMP-18)", definition.committee_refs)
        self.assertIn("(393)", definition.section_refs)
        
    def test_generate_context_tags(self):
        definition = Definition(
            term="Circuit Breaker",
            definition="A device designed to open and close a circuit.",
            alternative_terms=["Breaker"],
            committee_refs=["(CMP-10)"],
            section_refs=["(240)"],
            informational_notes=[]
        )
        
        tags = self.processor.generate_context_tags(definition)
        
        self.assertIn("equipment", tags)
        self.assertIn("protection", tags)
        self.assertIn("Section_240", tags)

if __name__ == '__main__':
    unittest.main()