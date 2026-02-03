/**
 * @description Test class for FormBuilderController
 */
@isTest
private class FormBuilderControllerTest {
    
    private static final String TEST_FORM_NAME = 'Test Form';
    private static final String TEST_FORM_DESC = 'Test Description';
    
    /**
     * @description Creates test form definition JSON
     */
    private static String getTestFormDefinition() {
        Map<String, Object> formDef = new Map<String, Object>{
            'name' => TEST_FORM_NAME,
            'description' => TEST_FORM_DESC,
            'elements' => new List<Object>{
                new Map<String, Object>{
                    'id' => 'element_1',
                    'type' => 'text',
                    'label' => 'First Name',
                    'apiName' => 'first_name',
                    'required' => true
                },
                new Map<String, Object>{
                    'id' => 'element_2',
                    'type' => 'email',
                    'label' => 'Email',
                    'apiName' => 'email',
                    'required' => true
                }
            }
        };
        return JSON.serialize(formDef);
    }
    
    /**
     * @description Creates test form submission data JSON
     */
    private static String getTestSubmissionData() {
        Map<String, Object> submissionData = new Map<String, Object>{
            'first_name' => 'John',
            'email' => 'john@example.com',
            'submittedAt' => System.now().format()
        };
        return JSON.serialize(submissionData);
    }
    
    /**
     * @description Setup test data
     */
    @TestSetup
    static void setupTestData() {
        Form_Definition__c testForm = new Form_Definition__c(
            Name = TEST_FORM_NAME,
            Description__c = TEST_FORM_DESC,
            Form_Definition__c = getTestFormDefinition(),
            Is_Active__c = true,
            Version__c = 1
        );
        insert testForm;
    }
    
    /**
     * @description Test saving a new form definition
     */
    @isTest
    static void testSaveNewFormDefinition() {
        Test.startTest();
        
        Id formId = FormBuilderController.saveFormDefinition(
            null,
            'New Form',
            'New Description',
            getTestFormDefinition()
        );
        
        Test.stopTest();
        
        System.assertNotEquals(null, formId, 'Form ID should not be null');
        
        Form_Definition__c savedForm = [
            SELECT Id, Name, Description__c, Form_Definition__c
            FROM Form_Definition__c
            WHERE Id = :formId
        ];
        
        System.assertEquals('New Form', savedForm.Name, 'Form name should match');
        System.assertEquals('New Description', savedForm.Description__c, 'Description should match');
    }
    
    /**
     * @description Test updating an existing form definition
     */
    @isTest
    static void testUpdateFormDefinition() {
        Form_Definition__c existingForm = [
            SELECT Id FROM Form_Definition__c LIMIT 1
        ];
        
        Test.startTest();
        
        Id formId = FormBuilderController.saveFormDefinition(
            existingForm.Id,
            'Updated Form',
            'Updated Description',
            getTestFormDefinition()
        );
        
        Test.stopTest();
        
        System.assertEquals(existingForm.Id, formId, 'Form ID should match');
        
        Form_Definition__c updatedForm = [
            SELECT Id, Name, Description__c
            FROM Form_Definition__c
            WHERE Id = :formId
        ];
        
        System.assertEquals('Updated Form', updatedForm.Name, 'Form name should be updated');
    }
    
    /**
     * @description Test saving form with blank name
     */
    @isTest
    static void testSaveFormWithBlankName() {
        Test.startTest();
        
        try {
            FormBuilderController.saveFormDefinition(
                null,
                '',
                'Description',
                getTestFormDefinition()
            );
            System.assert(false, 'Expected exception was not thrown');
        } catch (AuraHandledException e) {
            System.assert(e.getMessage().contains('Form name is required'), 
                'Error message should indicate name is required');
        }
        
        Test.stopTest();
    }
    
    /**
     * @description Test getting form definition
     */
    @isTest
    static void testGetFormDefinition() {
        Form_Definition__c existingForm = [
            SELECT Id FROM Form_Definition__c LIMIT 1
        ];
        
        Test.startTest();
        
        Form_Definition__c form = FormBuilderController.getFormDefinition(existingForm.Id);
        
        Test.stopTest();
        
        System.assertNotEquals(null, form, 'Form should not be null');
        System.assertEquals(TEST_FORM_NAME, form.Name, 'Form name should match');
    }
    
    /**
     * @description Test getting form definition with null ID
     */
    @isTest
    static void testGetFormDefinitionWithNullId() {
        Test.startTest();
        
        try {
            FormBuilderController.getFormDefinition(null);
            System.assert(false, 'Expected exception was not thrown');
        } catch (AuraHandledException e) {
            System.assert(e.getMessage().contains('Form ID is required'), 
                'Error message should indicate ID is required');
        }
        
        Test.stopTest();
    }
    
    /**
     * @description Test getting active form definitions
     */
    @isTest
    static void testGetActiveFormDefinitions() {
        Test.startTest();
        
        List<Form_Definition__c> forms = FormBuilderController.getActiveFormDefinitions();
        
        Test.stopTest();
        
        System.assertNotEquals(0, forms.size(), 'Should return at least one form');
    }
    
    /**
     * @description Test submitting form data
     */
    @isTest
    static void testSubmitFormData() {
        Form_Definition__c existingForm = [
            SELECT Id FROM Form_Definition__c LIMIT 1
        ];
        
        Test.startTest();
        
        Id submissionId = FormBuilderController.submitFormData(
            existingForm.Id,
            getTestSubmissionData()
        );
        
        Test.stopTest();
        
        System.assertNotEquals(null, submissionId, 'Submission ID should not be null');
        
        Form_Submission__c submission = [
            SELECT Id, Status__c, Form_Definition__c
            FROM Form_Submission__c
            WHERE Id = :submissionId
        ];
        
        System.assertEquals('Submitted', submission.Status__c, 'Status should be Submitted');
        System.assertEquals(existingForm.Id, submission.Form_Definition__c, 
            'Form definition should match');
    }
    
    /**
     * @description Test submitting form data with null form ID
     */
    @isTest
    static void testSubmitFormDataWithNullId() {
        Test.startTest();
        
        try {
            FormBuilderController.submitFormData(null, getTestSubmissionData());
            System.assert(false, 'Expected exception was not thrown');
        } catch (AuraHandledException e) {
            System.assert(e.getMessage().contains('Form definition ID is required'), 
                'Error message should indicate ID is required');
        }
        
        Test.stopTest();
    }
    
    /**
     * @description Test getting form submissions
     */
    @isTest
    static void testGetFormSubmissions() {
        Form_Definition__c existingForm = [
            SELECT Id FROM Form_Definition__c LIMIT 1
        ];
        
        // Create test submission
        Form_Submission__c submission = new Form_Submission__c(
            Form_Definition__c = existingForm.Id,
            Submission_Data__c = getTestSubmissionData(),
            Submitted_Date__c = System.now(),
            Status__c = 'Submitted'
        );
        insert submission;
        
        Test.startTest();
        
        List<Form_Submission__c> submissions = FormBuilderController.getFormSubmissions(existingForm.Id);
        
        Test.stopTest();
        
        System.assertNotEquals(0, submissions.size(), 'Should return at least one submission');
    }
    
    /**
     * @description Test getting single form submission
     */
    @isTest
    static void testGetFormSubmission() {
        Form_Definition__c existingForm = [
            SELECT Id FROM Form_Definition__c LIMIT 1
        ];
        
        Form_Submission__c submission = new Form_Submission__c(
            Form_Definition__c = existingForm.Id,
            Submission_Data__c = getTestSubmissionData(),
            Submitted_Date__c = System.now(),
            Status__c = 'Submitted'
        );
        insert submission;
        
        Test.startTest();
        
        Form_Submission__c result = FormBuilderController.getFormSubmission(submission.Id);
        
        Test.stopTest();
        
        System.assertNotEquals(null, result, 'Submission should not be null');
        System.assertEquals(submission.Id, result.Id, 'Submission ID should match');
    }
    
    /**
     * @description Test deleting form definition
     */
    @isTest
    static void testDeleteFormDefinition() {
        Form_Definition__c formToDelete = new Form_Definition__c(
            Name = 'Form To Delete',
            Description__c = 'Will be deleted',
            Form_Definition__c = getTestFormDefinition(),
            Is_Active__c = true,
            Version__c = 1
        );
        insert formToDelete;
        
        // Add a submission
        Form_Submission__c submission = new Form_Submission__c(
            Form_Definition__c = formToDelete.Id,
            Submission_Data__c = getTestSubmissionData(),
            Submitted_Date__c = System.now(),
            Status__c = 'Submitted'
        );
        insert submission;
        
        Test.startTest();
        
        FormBuilderController.deleteFormDefinition(formToDelete.Id);
        
        Test.stopTest();
        
        List<Form_Definition__c> forms = [
            SELECT Id FROM Form_Definition__c WHERE Id = :formToDelete.Id
        ];
        System.assertEquals(0, forms.size(), 'Form should be deleted');
        
        List<Form_Submission__c> submissions = [
            SELECT Id FROM Form_Submission__c WHERE Form_Definition__c = :formToDelete.Id
        ];
        System.assertEquals(0, submissions.size(), 'Submissions should be deleted');
    }
    
    /**
     * @description Test updating form status
     */
    @isTest
    static void testUpdateFormStatus() {
        Form_Definition__c existingForm = [
            SELECT Id, Is_Active__c FROM Form_Definition__c LIMIT 1
        ];
        
        Test.startTest();
        
        FormBuilderController.updateFormStatus(existingForm.Id, false);
        
        Test.stopTest();
        
        Form_Definition__c updatedForm = [
            SELECT Is_Active__c FROM Form_Definition__c WHERE Id = :existingForm.Id
        ];
        System.assertEquals(false, updatedForm.Is_Active__c, 'Form should be inactive');
    }
    
    /**
     * @description Test duplicating form
     */
    @isTest
    static void testDuplicateForm() {
        Form_Definition__c existingForm = [
            SELECT Id, Name FROM Form_Definition__c LIMIT 1
        ];
        
        Test.startTest();
        
        Id newFormId = FormBuilderController.duplicateForm(existingForm.Id);
        
        Test.stopTest();
        
        System.assertNotEquals(null, newFormId, 'New form ID should not be null');
        System.assertNotEquals(existingForm.Id, newFormId, 'New form should have different ID');
        
        Form_Definition__c newForm = [
            SELECT Name, Is_Active__c FROM Form_Definition__c WHERE Id = :newFormId
        ];
        System.assert(newForm.Name.contains('(Copy)'), 'New form name should contain (Copy)');
        System.assertEquals(false, newForm.Is_Active__c, 'New form should be inactive');
    }
}
