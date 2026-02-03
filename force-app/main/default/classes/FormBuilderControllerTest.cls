/**
 * @description Test class for FormBuilderController
 */
@isTest
private class FormBuilderControllerTest {
    
    @TestSetup
    static void setupTestData() {
        // Create test form definition
        Form_Definition__c testForm = new Form_Definition__c();
        testForm.Name = 'Test Form';
        testForm.Description__c = 'A test form';
        testForm.Form_JSON__c = '{"fields":[{"id":"field1","type":"text","label":"First Name","required":true,"targetField":"FirstName"},{"id":"field2","type":"email","label":"Email","required":false,"targetField":"Email"}]}';
        testForm.Is_Active__c = true;
        testForm.Target_Object__c = 'Contact';
        insert testForm;
        
        // Create test form response
        Form_Response__c testResponse = new Form_Response__c();
        testResponse.Form_Definition__c = testForm.Id;
        testResponse.Response_JSON__c = '{"field1":"John","field2":"john@test.com"}';
        testResponse.Submission_Date__c = DateTime.now();
        testResponse.Submitted_By__c = UserInfo.getUserId();
        insert testResponse;
    }
    
    @isTest
    static void testSaveFormDefinition_New() {
        String formJson = '{"fields":[{"id":"f1","type":"text","label":"Name"}]}';
        
        Test.startTest();
        String formId = FormBuilderController.saveFormDefinition(
            null,
            'New Test Form',
            'Description',
            formJson,
            'Account',
            true
        );
        Test.stopTest();
        
        System.assertNotEquals(null, formId, 'Form ID should not be null');
        
        Form_Definition__c savedForm = [SELECT Name, Description__c FROM Form_Definition__c WHERE Id = :formId];
        System.assertEquals('New Test Form', savedForm.Name, 'Form name should match');
    }
    
    @isTest
    static void testSaveFormDefinition_Update() {
        Form_Definition__c existingForm = [SELECT Id FROM Form_Definition__c LIMIT 1];
        String formJson = '{"fields":[{"id":"f1","type":"text","label":"Updated Name"}]}';
        
        Test.startTest();
        String formId = FormBuilderController.saveFormDefinition(
            existingForm.Id,
            'Updated Form Name',
            'Updated Description',
            formJson,
            'Lead',
            false
        );
        Test.stopTest();
        
        System.assertEquals(existingForm.Id, formId, 'Form ID should match existing');
        
        Form_Definition__c updatedForm = [SELECT Name, Is_Active__c FROM Form_Definition__c WHERE Id = :formId];
        System.assertEquals('Updated Form Name', updatedForm.Name, 'Form name should be updated');
        System.assertEquals(false, updatedForm.Is_Active__c, 'Form should be inactive');
    }
    
    @isTest
    static void testGetFormDefinition() {
        Form_Definition__c testForm = [SELECT Id FROM Form_Definition__c LIMIT 1];
        
        Test.startTest();
        Form_Definition__c result = FormBuilderController.getFormDefinition(testForm.Id);
        Test.stopTest();
        
        System.assertNotEquals(null, result, 'Result should not be null');
        System.assertEquals('Test Form', result.Name, 'Form name should match');
    }
    
    @isTest
    static void testGetAllFormDefinitions() {
        Test.startTest();
        List<Form_Definition__c> results = FormBuilderController.getAllFormDefinitions();
        Test.stopTest();
        
        System.assert(results.size() > 0, 'Should return at least one form');
    }
    
    @isTest
    static void testGetActiveFormDefinitions() {
        Test.startTest();
        List<Form_Definition__c> results = FormBuilderController.getActiveFormDefinitions();
        Test.stopTest();
        
        System.assert(results.size() > 0, 'Should return at least one active form');
        for (Form_Definition__c form : results) {
            System.assertEquals(true, form.Is_Active__c, 'All forms should be active');
        }
    }
    
    @isTest
    static void testDeleteFormDefinition() {
        // Create a form to delete
        Form_Definition__c formToDelete = new Form_Definition__c();
        formToDelete.Name = 'Form to Delete';
        formToDelete.Form_JSON__c = '{}';
        insert formToDelete;
        
        Test.startTest();
        FormBuilderController.deleteFormDefinition(formToDelete.Id);
        Test.stopTest();
        
        List<Form_Definition__c> remainingForms = [SELECT Id FROM Form_Definition__c WHERE Id = :formToDelete.Id];
        System.assertEquals(0, remainingForms.size(), 'Form should be deleted');
    }
    
    @isTest
    static void testSubmitFormResponse() {
        Form_Definition__c testForm = [SELECT Id FROM Form_Definition__c WHERE Is_Active__c = true LIMIT 1];
        String responseJson = '{"field1":"Jane","field2":"jane@test.com"}';
        
        // Create an account first for target object test
        Account testAccount = new Account(Name = 'Test Account');
        insert testAccount;
        
        Test.startTest();
        String responseId = FormBuilderController.submitFormResponse(testForm.Id, responseJson);
        Test.stopTest();
        
        System.assertNotEquals(null, responseId, 'Response ID should not be null');
        
        Form_Response__c savedResponse = [SELECT Response_JSON__c FROM Form_Response__c WHERE Id = :responseId];
        System.assertEquals(responseJson, savedResponse.Response_JSON__c, 'Response JSON should match');
    }
    
    @isTest
    static void testSubmitFormResponse_InactiveForm() {
        Form_Definition__c testForm = [SELECT Id FROM Form_Definition__c LIMIT 1];
        testForm.Is_Active__c = false;
        update testForm;
        
        String responseJson = '{"field1":"Test"}';
        
        Test.startTest();
        try {
            FormBuilderController.submitFormResponse(testForm.Id, responseJson);
            System.assert(false, 'Should have thrown an exception');
        } catch (AuraHandledException e) {
            System.assert(e.getMessage().contains('no longer active'), 'Should indicate form is inactive');
        }
        Test.stopTest();
    }
    
    @isTest
    static void testGetFormResponses() {
        Form_Definition__c testForm = [SELECT Id FROM Form_Definition__c LIMIT 1];
        
        Test.startTest();
        List<Form_Response__c> results = FormBuilderController.getFormResponses(testForm.Id);
        Test.stopTest();
        
        System.assert(results.size() > 0, 'Should return at least one response');
    }
    
    @isTest
    static void testGetAvailableObjects() {
        Test.startTest();
        List<Map<String, String>> results = FormBuilderController.getAvailableObjects();
        Test.stopTest();
        
        System.assert(results.size() > 0, 'Should return available objects');
    }
    
    @isTest
    static void testGetObjectFields() {
        Test.startTest();
        List<Map<String, String>> results = FormBuilderController.getObjectFields('Account');
        Test.stopTest();
        
        System.assert(results.size() > 0, 'Should return Account fields');
    }
    
    @isTest
    static void testGetObjectFields_InvalidObject() {
        Test.startTest();
        try {
            FormBuilderController.getObjectFields('InvalidObject__xyz');
            System.assert(false, 'Should have thrown an exception');
        } catch (AuraHandledException e) {
            System.assert(e.getMessage().contains('not found'), 'Should indicate object not found');
        }
        Test.stopTest();
    }
    
    @isTest
    static void testDuplicateFormDefinition() {
        Form_Definition__c testForm = [SELECT Id FROM Form_Definition__c LIMIT 1];
        
        Test.startTest();
        String newFormId = FormBuilderController.duplicateFormDefinition(testForm.Id, 'Duplicated Form');
        Test.stopTest();
        
        System.assertNotEquals(null, newFormId, 'New form ID should not be null');
        System.assertNotEquals(testForm.Id, newFormId, 'New form should have different ID');
        
        Form_Definition__c duplicatedForm = [SELECT Name, Is_Active__c FROM Form_Definition__c WHERE Id = :newFormId];
        System.assertEquals('Duplicated Form', duplicatedForm.Name, 'Name should match');
        System.assertEquals(false, duplicatedForm.Is_Active__c, 'Duplicated form should be inactive');
    }
    
    @isTest
    static void testDuplicateFormDefinition_DefaultName() {
        Form_Definition__c testForm = [SELECT Id, Name FROM Form_Definition__c LIMIT 1];
        
        Test.startTest();
        String newFormId = FormBuilderController.duplicateFormDefinition(testForm.Id, null);
        Test.stopTest();
        
        Form_Definition__c duplicatedForm = [SELECT Name FROM Form_Definition__c WHERE Id = :newFormId];
        System.assert(duplicatedForm.Name.contains('(Copy)'), 'Name should contain (Copy)');
    }
}
