import unittest

from backend.resume.parser import extract_resume


class ResumeParserTests(unittest.TestCase):
    def test_extracts_contact_sections_and_skills(self):
        resume = """Asha Kumar
asha@example.com | +91 98765 43210 | https://github.com/asha

EDUCATION
B.Tech Computer Science, State University, 2024

SKILLS
Python, Flask, MongoDB, OpenCV, SQL, React

PROJECTS
- Smart Attendance System using Python, Flask, MongoDB and OpenCV
- Portfolio website using React

EXPERIENCE
Software Intern at Acme, 2023

CERTIFICATIONS
AWS Cloud Practitioner
"""
        profile = extract_resume(resume)
        self.assertEqual(profile["contact"]["emails"], ["asha@example.com"])
        self.assertIn("+91 98765 43210", profile["contact"]["phones"])
        self.assertIn("python", profile["skills"])
        self.assertIn("flask", profile["skills"])
        self.assertEqual(len(profile["projects"]), 2)
        self.assertTrue(profile["education"])
        self.assertTrue(profile["experience"])
        self.assertIn("2024", profile["years"])

    def test_rejects_empty_resume(self):
        with self.assertRaises(ValueError):
            extract_resume(" ")


if __name__ == "__main__":
    unittest.main()
