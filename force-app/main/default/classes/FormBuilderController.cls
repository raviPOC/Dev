/**
 * @description Apex controller for the Form Builder LWC component.
 * Handles CRUD operations for form definitions and form submissions.
 * @author Form Builder Team
 */
public with sharing class FormBuilderController {
    
    /**
     * @description Saves or updates a form definition
     * @param formId The ID of the form to update (null for new forms)
     * @param formName The name of the form
     * @param formDescription The description of the form
     * @param formDefinitionJson The JSON definition of the form
     * @return The ID of the saved form definition
     */
    @AuraEnabled
    public static Id saveFormDefinition(Id formId, String formName, String formDescription, String formDefinitionJson) {
        try {
            // Validate inputs
            if (String.isBlank(formName)) {
                throw new AuraHandledException('Form name is required');
            }
            if (String.isBlank(formDefinitionJson)) {
                throw new AuraHandledException('Form definition is required');
            }
            
            // Validate JSON
            try {
                JSON.deserializeUntyped(formDefinitionJson);
            } catch (JSONException e) {
                throw new AuraHandledException('Invalid form definition JSON: ' + e.getMessage());
            }
            
            Form_Definition__c formDef;
            
            if (formId != null) {
                // Update existing form
                formDef = [
                    SELECT Id, Name, Description__c, Form_Definition__c, Is_Active__c
                    FROM Form_Definition__c
                    WHERE Id = :formId
                    WITH SECURITY_ENFORCED
                    LIMIT 1
                ];
                formDef.Name = formName;
                formDef.Description__c = formDescription;
                formDef.Form_Definition__c = formDefinitionJson;
                formDef.Last_Modified_Date__c = System.now();
            } else {
                // Create new form
                formDef = new Form_Definition__c(
                    Name = formName,
                    Description__c = formDescription,
                    Form_Definition__c = formDefinitionJson,
                    Is_Active__c = true,
                    Version__c = 1
                );
            }
            
            // Perform DML
            if (formId != null) {
                update formDef;
            } else {
                insert formDef;
            }
            
            return formDef.Id;
            
        } catch (Exception e) {
            throw new AuraHandledException('Error saving form: ' + e.getMessage());
        }
    }
    
    /**
     * @description Retrieves a form definition by ID
     * @param formId The ID of the form to retrieve
     * @return The form definition record
     */
    @AuraEnabled(cacheable=true)
    public static Form_Definition__c getFormDefinition(Id formId) {
        try {
            if (formId == null) {
                throw new AuraHandledException('Form ID is required');
            }
            
            List<Form_Definition__c> forms = [
                SELECT Id, Name, Description__c, Form_Definition__c, Is_Active__c, 
                       Version__c, CreatedDate, LastModifiedDate
                FROM Form_Definition__c
                WHERE Id = :formId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
            
            if (forms.isEmpty()) {
                throw new AuraHandledException('Form not found');
            }
            
            return forms[0];
            
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving form: ' + e.getMessage());
        }
    }
    
    /**
     * @description Retrieves all active form definitions
     * @return List of active form definitions
     */
    @AuraEnabled(cacheable=true)
    public static List<Form_Definition__c> getActiveFormDefinitions() {
        try {
            return [
                SELECT Id, Name, Description__c, Is_Active__c, Version__c, 
                       CreatedDate, LastModifiedDate
                FROM Form_Definition__c
                WHERE Is_Active__c = true
                WITH SECURITY_ENFORCED
                ORDER BY Name ASC
                LIMIT 100
            ];
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving forms: ' + e.getMessage());
        }
    }
    
    /**
     * @description Submits form data
     * @param formDefinitionId The ID of the form definition
     * @param formDataJson The JSON data submitted by the user
     * @return The ID of the created form submission
     */
    @AuraEnabled
    public static Id submitFormData(Id formDefinitionId, String formDataJson) {
        try {
            // Validate inputs
            if (formDefinitionId == null) {
                throw new AuraHandledException('Form definition ID is required');
            }
            if (String.isBlank(formDataJson)) {
                throw new AuraHandledException('Form data is required');
            }
            
            // Validate JSON
            try {
                JSON.deserializeUntyped(formDataJson);
            } catch (JSONException e) {
                throw new AuraHandledException('Invalid form data JSON: ' + e.getMessage());
            }
            
            // Verify form exists
            List<Form_Definition__c> forms = [
                SELECT Id, Name
                FROM Form_Definition__c
                WHERE Id = :formDefinitionId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
            
            if (forms.isEmpty()) {
                throw new AuraHandledException('Form definition not found');
            }
            
            // Create form submission
            Form_Submission__c submission = new Form_Submission__c(
                Form_Definition__c = formDefinitionId,
                Submission_Data__c = formDataJson,
                Submitted_Date__c = System.now(),
                Status__c = 'Submitted'
            );
            
            insert submission;
            
            return submission.Id;
            
        } catch (Exception e) {
            throw new AuraHandledException('Error submitting form: ' + e.getMessage());
        }
    }
    
    /**
     * @description Retrieves form submissions for a specific form
     * @param formDefinitionId The ID of the form definition
     * @return List of form submissions
     */
    @AuraEnabled(cacheable=true)
    public static List<Form_Submission__c> getFormSubmissions(Id formDefinitionId) {
        try {
            if (formDefinitionId == null) {
                throw new AuraHandledException('Form definition ID is required');
            }
            
            return [
                SELECT Id, Name, Submission_Data__c, Submitted_Date__c, Status__c,
                       CreatedById, CreatedBy.Name, CreatedDate
                FROM Form_Submission__c
                WHERE Form_Definition__c = :formDefinitionId
                WITH SECURITY_ENFORCED
                ORDER BY CreatedDate DESC
                LIMIT 200
            ];
            
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving submissions: ' + e.getMessage());
        }
    }
    
    /**
     * @description Retrieves a single form submission by ID
     * @param submissionId The ID of the submission
     * @return The form submission record
     */
    @AuraEnabled(cacheable=true)
    public static Form_Submission__c getFormSubmission(Id submissionId) {
        try {
            if (submissionId == null) {
                throw new AuraHandledException('Submission ID is required');
            }
            
            List<Form_Submission__c> submissions = [
                SELECT Id, Name, Submission_Data__c, Submitted_Date__c, Status__c,
                       Form_Definition__c, Form_Definition__r.Name,
                       CreatedById, CreatedBy.Name, CreatedDate
                FROM Form_Submission__c
                WHERE Id = :submissionId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
            
            if (submissions.isEmpty()) {
                throw new AuraHandledException('Submission not found');
            }
            
            return submissions[0];
            
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving submission: ' + e.getMessage());
        }
    }
    
    /**
     * @description Deletes a form definition and its submissions
     * @param formId The ID of the form to delete
     */
    @AuraEnabled
    public static void deleteFormDefinition(Id formId) {
        try {
            if (formId == null) {
                throw new AuraHandledException('Form ID is required');
            }
            
            // Delete associated submissions first
            List<Form_Submission__c> submissions = [
                SELECT Id
                FROM Form_Submission__c
                WHERE Form_Definition__c = :formId
                WITH SECURITY_ENFORCED
            ];
            
            if (!submissions.isEmpty()) {
                delete submissions;
            }
            
            // Delete form definition
            Form_Definition__c formDef = [
                SELECT Id
                FROM Form_Definition__c
                WHERE Id = :formId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
            
            delete formDef;
            
        } catch (Exception e) {
            throw new AuraHandledException('Error deleting form: ' + e.getMessage());
        }
    }
    
    /**
     * @description Updates the active status of a form
     * @param formId The ID of the form
     * @param isActive The new active status
     */
    @AuraEnabled
    public static void updateFormStatus(Id formId, Boolean isActive) {
        try {
            if (formId == null) {
                throw new AuraHandledException('Form ID is required');
            }
            
            Form_Definition__c formDef = [
                SELECT Id, Is_Active__c
                FROM Form_Definition__c
                WHERE Id = :formId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
            
            formDef.Is_Active__c = isActive;
            update formDef;
            
        } catch (Exception e) {
            throw new AuraHandledException('Error updating form status: ' + e.getMessage());
        }
    }
    
    /**
     * @description Duplicates an existing form
     * @param formId The ID of the form to duplicate
     * @return The ID of the new form
     */
    @AuraEnabled
    public static Id duplicateForm(Id formId) {
        try {
            if (formId == null) {
                throw new AuraHandledException('Form ID is required');
            }
            
            Form_Definition__c originalForm = [
                SELECT Id, Name, Description__c, Form_Definition__c
                FROM Form_Definition__c
                WHERE Id = :formId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
            
            Form_Definition__c newForm = new Form_Definition__c(
                Name = originalForm.Name + ' (Copy)',
                Description__c = originalForm.Description__c,
                Form_Definition__c = originalForm.Form_Definition__c,
                Is_Active__c = false,
                Version__c = 1
            );
            
            insert newForm;
            
            return newForm.Id;
            
        } catch (Exception e) {
            throw new AuraHandledException('Error duplicating form: ' + e.getMessage());
        }
    }
}
