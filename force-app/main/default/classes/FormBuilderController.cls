/**
 * @description Controller class for the LWC Form Builder component
 * Handles CRUD operations for form definitions and form responses
 * @author Form Builder Team
 */
public with sharing class FormBuilderController {
    
    /**
     * @description Saves a new form definition or updates an existing one
     * @param formId Optional ID of existing form to update
     * @param formName Name of the form
     * @param formDescription Description of the form
     * @param formJson JSON string containing the form structure
     * @param targetObject Optional API name of target object for data storage
     * @param isActive Whether the form is active
     * @return The ID of the saved form definition
     */
    @AuraEnabled
    public static String saveFormDefinition(
        String formId,
        String formName,
        String formDescription,
        String formJson,
        String targetObject,
        Boolean isActive
    ) {
        try {
            Form_Definition__c formDef;
            
            if (String.isNotBlank(formId)) {
                // Update existing form
                formDef = [
                    SELECT Id, Name, Description__c, Form_JSON__c, Target_Object__c, Is_Active__c
                    FROM Form_Definition__c
                    WHERE Id = :formId
                    WITH SECURITY_ENFORCED
                    LIMIT 1
                ];
            } else {
                // Create new form
                formDef = new Form_Definition__c();
            }
            
            formDef.Name = formName;
            formDef.Description__c = formDescription;
            formDef.Form_JSON__c = formJson;
            formDef.Target_Object__c = targetObject;
            formDef.Is_Active__c = isActive != null ? isActive : true;
            
            upsert formDef;
            
            return formDef.Id;
        } catch (Exception e) {
            throw new AuraHandledException('Error saving form: ' + e.getMessage());
        }
    }
    
    /**
     * @description Retrieves a form definition by ID
     * @param formId ID of the form definition
     * @return The form definition record
     */
    @AuraEnabled(cacheable=true)
    public static Form_Definition__c getFormDefinition(String formId) {
        try {
            return [
                SELECT Id, Name, Description__c, Form_JSON__c, Target_Object__c, Is_Active__c,
                       CreatedDate, LastModifiedDate, CreatedById, LastModifiedById
                FROM Form_Definition__c
                WHERE Id = :formId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving form: ' + e.getMessage());
        }
    }
    
    /**
     * @description Retrieves all form definitions
     * @return List of form definitions
     */
    @AuraEnabled(cacheable=true)
    public static List<Form_Definition__c> getAllFormDefinitions() {
        try {
            return [
                SELECT Id, Name, Description__c, Is_Active__c, Target_Object__c,
                       CreatedDate, LastModifiedDate
                FROM Form_Definition__c
                WITH SECURITY_ENFORCED
                ORDER BY LastModifiedDate DESC
                LIMIT 100
            ];
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving forms: ' + e.getMessage());
        }
    }
    
    /**
     * @description Retrieves active form definitions only
     * @return List of active form definitions
     */
    @AuraEnabled(cacheable=true)
    public static List<Form_Definition__c> getActiveFormDefinitions() {
        try {
            return [
                SELECT Id, Name, Description__c, Form_JSON__c, Target_Object__c,
                       CreatedDate, LastModifiedDate
                FROM Form_Definition__c
                WHERE Is_Active__c = true
                WITH SECURITY_ENFORCED
                ORDER BY Name ASC
                LIMIT 100
            ];
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving active forms: ' + e.getMessage());
        }
    }
    
    /**
     * @description Deletes a form definition
     * @param formId ID of the form to delete
     */
    @AuraEnabled
    public static void deleteFormDefinition(String formId) {
        try {
            Form_Definition__c formDef = [
                SELECT Id FROM Form_Definition__c
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
     * @description Submits a form response and optionally creates a target record
     * @param formId ID of the form definition
     * @param responseJson JSON string containing form field values
     * @return The ID of the created form response
     */
    @AuraEnabled
    public static String submitFormResponse(String formId, String responseJson) {
        try {
            // Get form definition
            Form_Definition__c formDef = [
                SELECT Id, Name, Target_Object__c, Form_JSON__c, Is_Active__c
                FROM Form_Definition__c
                WHERE Id = :formId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
            
            if (!formDef.Is_Active__c) {
                throw new AuraHandledException('This form is no longer active');
            }
            
            // Create form response record
            Form_Response__c response = new Form_Response__c();
            response.Form_Definition__c = formId;
            response.Response_JSON__c = responseJson;
            response.Submission_Date__c = DateTime.now();
            response.Submitted_By__c = UserInfo.getUserId();
            
            // If target object is specified, create record in that object
            if (String.isNotBlank(formDef.Target_Object__c)) {
                String targetRecordId = createTargetRecord(formDef.Target_Object__c, responseJson, formDef.Form_JSON__c);
                response.Target_Record_Id__c = targetRecordId;
            }
            
            insert response;
            
            return response.Id;
        } catch (Exception e) {
            throw new AuraHandledException('Error submitting form: ' + e.getMessage());
        }
    }
    
    /**
     * @description Creates a record in the target object using form response data
     * @param targetObject API name of the target object
     * @param responseJson JSON containing form field values
     * @param formJson JSON containing form structure with field mappings
     * @return ID of the created record
     */
    private static String createTargetRecord(String targetObject, String responseJson, String formJson) {
        try {
            // Parse form definition to get field mappings
            Map<String, Object> formStructure = (Map<String, Object>) JSON.deserializeUntyped(formJson);
            List<Object> fields = (List<Object>) formStructure.get('fields');
            
            // Parse response values
            Map<String, Object> responseValues = (Map<String, Object>) JSON.deserializeUntyped(responseJson);
            
            // Create SObject dynamically
            Schema.SObjectType targetType = Schema.getGlobalDescribe().get(targetObject);
            if (targetType == null) {
                throw new AuraHandledException('Invalid target object: ' + targetObject);
            }
            
            SObject targetRecord = targetType.newSObject();
            
            // Map response values to target object fields
            for (Object fieldObj : fields) {
                Map<String, Object> field = (Map<String, Object>) fieldObj;
                String fieldId = (String) field.get('id');
                String targetField = (String) field.get('targetField');
                
                if (String.isNotBlank(targetField) && responseValues.containsKey(fieldId)) {
                    Object value = responseValues.get(fieldId);
                    setFieldValue(targetRecord, targetField, value, targetType);
                }
            }
            
            insert targetRecord;
            return targetRecord.Id;
        } catch (Exception e) {
            throw new AuraHandledException('Error creating target record: ' + e.getMessage());
        }
    }
    
    /**
     * @description Sets a field value on an SObject with proper type conversion
     * @param record The SObject record
     * @param fieldName API name of the field
     * @param value The value to set
     * @param objectType Schema type of the object
     */
    private static void setFieldValue(SObject record, String fieldName, Object value, Schema.SObjectType objectType) {
        try {
            Schema.DescribeFieldResult fieldDescribe = objectType.getDescribe().fields.getMap().get(fieldName).getDescribe();
            Schema.DisplayType fieldType = fieldDescribe.getType();
            
            if (value == null) {
                record.put(fieldName, null);
                return;
            }
            
            // Convert value based on field type
            switch on fieldType {
                when BOOLEAN {
                    record.put(fieldName, Boolean.valueOf(String.valueOf(value)));
                }
                when DATE {
                    record.put(fieldName, Date.valueOf(String.valueOf(value)));
                }
                when DATETIME {
                    record.put(fieldName, DateTime.valueOf(String.valueOf(value)));
                }
                when DOUBLE, CURRENCY, PERCENT {
                    record.put(fieldName, Double.valueOf(String.valueOf(value)));
                }
                when INTEGER {
                    record.put(fieldName, Integer.valueOf(String.valueOf(value)));
                }
                when else {
                    record.put(fieldName, String.valueOf(value));
                }
            }
        } catch (Exception e) {
            // Log error but continue with other fields
            System.debug('Error setting field ' + fieldName + ': ' + e.getMessage());
        }
    }
    
    /**
     * @description Retrieves form responses for a specific form
     * @param formId ID of the form definition
     * @return List of form responses
     */
    @AuraEnabled(cacheable=true)
    public static List<Form_Response__c> getFormResponses(String formId) {
        try {
            return [
                SELECT Id, Name, Response_JSON__c, Submission_Date__c, 
                       Submitted_By__c, Submitted_By__r.Name, Target_Record_Id__c,
                       Form_Definition__c, Form_Definition__r.Name
                FROM Form_Response__c
                WHERE Form_Definition__c = :formId
                WITH SECURITY_ENFORCED
                ORDER BY Submission_Date__c DESC
                LIMIT 200
            ];
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving responses: ' + e.getMessage());
        }
    }
    
    /**
     * @description Gets available objects for target object selection
     * @return List of object options
     */
    @AuraEnabled(cacheable=true)
    public static List<Map<String, String>> getAvailableObjects() {
        List<Map<String, String>> objects = new List<Map<String, String>>();
        
        try {
            Map<String, Schema.SObjectType> globalDescribe = Schema.getGlobalDescribe();
            
            for (String objectName : globalDescribe.keySet()) {
                Schema.DescribeSObjectResult objDescribe = globalDescribe.get(objectName).getDescribe();
                
                // Only include createable objects
                if (objDescribe.isCreateable() && objDescribe.isAccessible() && !objDescribe.isCustomSetting()) {
                    Map<String, String> objOption = new Map<String, String>();
                    objOption.put('label', objDescribe.getLabel());
                    objOption.put('value', objectName);
                    objects.add(objOption);
                }
            }
            
            // Sort by label
            objects.sort();
            
            return objects;
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving objects: ' + e.getMessage());
        }
    }
    
    /**
     * @description Gets available fields for a specific object
     * @param objectName API name of the object
     * @return List of field options
     */
    @AuraEnabled(cacheable=true)
    public static List<Map<String, String>> getObjectFields(String objectName) {
        List<Map<String, String>> fields = new List<Map<String, String>>();
        
        try {
            Schema.SObjectType objectType = Schema.getGlobalDescribe().get(objectName);
            if (objectType == null) {
                throw new AuraHandledException('Object not found: ' + objectName);
            }
            
            Map<String, Schema.SObjectField> fieldMap = objectType.getDescribe().fields.getMap();
            
            for (String fieldName : fieldMap.keySet()) {
                Schema.DescribeFieldResult fieldDescribe = fieldMap.get(fieldName).getDescribe();
                
                // Only include createable and updatable fields
                if (fieldDescribe.isCreateable() && fieldDescribe.isAccessible()) {
                    Map<String, String> fieldOption = new Map<String, String>();
                    fieldOption.put('label', fieldDescribe.getLabel());
                    fieldOption.put('value', fieldName);
                    fieldOption.put('type', String.valueOf(fieldDescribe.getType()));
                    fieldOption.put('required', String.valueOf(!fieldDescribe.isNillable()));
                    fields.add(fieldOption);
                }
            }
            
            return fields;
        } catch (Exception e) {
            throw new AuraHandledException('Error retrieving fields: ' + e.getMessage());
        }
    }
    
    /**
     * @description Duplicates an existing form definition
     * @param formId ID of the form to duplicate
     * @param newName Name for the duplicated form
     * @return ID of the new form definition
     */
    @AuraEnabled
    public static String duplicateFormDefinition(String formId, String newName) {
        try {
            Form_Definition__c originalForm = [
                SELECT Id, Name, Description__c, Form_JSON__c, Target_Object__c
                FROM Form_Definition__c
                WHERE Id = :formId
                WITH SECURITY_ENFORCED
                LIMIT 1
            ];
            
            Form_Definition__c newForm = new Form_Definition__c();
            newForm.Name = String.isNotBlank(newName) ? newName : originalForm.Name + ' (Copy)';
            newForm.Description__c = originalForm.Description__c;
            newForm.Form_JSON__c = originalForm.Form_JSON__c;
            newForm.Target_Object__c = originalForm.Target_Object__c;
            newForm.Is_Active__c = false; // New copies are inactive by default
            
            insert newForm;
            
            return newForm.Id;
        } catch (Exception e) {
            throw new AuraHandledException('Error duplicating form: ' + e.getMessage());
        }
    }
}
