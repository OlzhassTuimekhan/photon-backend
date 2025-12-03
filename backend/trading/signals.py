from django.db.models.signals import post_save
from django.dispatch import receiver
from trading.models import AgentStatus, User


@receiver(post_save, sender=User)
def create_default_agent_statuses(sender, instance, created, **kwargs):
    """Создаем статусы агентов при создании пользователя"""
    if created:
        for agent_type, _ in AgentStatus.AGENT_TYPES:
            AgentStatus.objects.get_or_create(
                user=instance,
                agent_type=agent_type,
                defaults={"status": "IDLE"},
            )

